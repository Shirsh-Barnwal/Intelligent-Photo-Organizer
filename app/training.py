"""
app/training.py — Train a DINOv2 (multi-scale) adapter aligned with text embeddings.

Architecture & Workflow:
  - Multi-scale DINOv2: CLS (768) + spatial pooling (768) = 1536-d visual representation
  - Adapter: 3-layer MLP projecting 1536-d -> 384-d text embedding space
  - Loss: CombinedLoss (0.7 * CosineEmbeddingLoss + 0.3 * MSELoss + L2 regularization)
  - Scheduler: OneCycleLR with warmup and cosine decay
  - Stability: EMA (Exponential Moving Average) of weights
  - Performance: Automatic VRAM optimization, mixed precision AMP, LMDB database caching
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import io
import json
import logging
import math
import os
import random as _rng
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from .caching import clean_rel_path, configured_lmdb_dir, path_key
from .utils import format_size, get_paths

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import lmdb
except Exception:
    lmdb = None

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_PATHS = get_paths()
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Default Training Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir": str(PROJECT_PATHS.data_dir),
    "base_weights_dir": str(PROJECT_PATHS.weights_dir),
    "cache_file": "text_emb_cache.pt",
    "dino_path": str(PROJECT_PATHS.dinov2_model_path),
    "text_model": str(PROJECT_PATHS.all_minilm_path),
    "input_dim": 1536,
    "output_dim": 384,
    "hidden_dim": 512,
    "dropout": 0.3,
    "batch_size": 32,
    "grad_accum_steps": 2,
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "num_workers": 4,
    "train_max_side": 1024,
    "augment_mode": "light",
    "lmdb_dir": str(configured_lmdb_dir()),
    "lmdb_map_headroom": 2.2,
    "cosine_loss_weight": 0.7,
    "mse_loss_weight": 0.3,
    "l2_lambda": 1e-3,
    "val_split": 0.20,
    "early_stopping_patience": 5,
    "ema_decay": 0.995,
    "label_noise": 0.02,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loader Supporting Flat, Detailed, & txt Annotations
# ─────────────────────────────────────────────────────────────────────────────

class ImageTextDataset(Dataset):
    """Loads image-text pairs from flat JSON, detailed metadata, or .txt files with LMDB support."""

    IMG_KEYS = ("image_path", "image", "file_path", "path", "img", "filename", "source")
    TXT_KEYS = ("description", "caption", "text", "label", "annotation")
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

    def __init__(
        self,
        data_dir: str,
        text_model_name: str,
        cache_file: str,
        device="cpu",
        lmdb_dir: str | None = None,
        lmdb_mode: str = "auto",
        rebuild_lmdb: bool = False,
        metadata_source: str = "auto",
        min_tag_score: float = 0.1,
        stop_file: str | None = None,
        train_max_side: int = 1024,
        augment_mode: str = "light",
    ):
        self.data_dir = data_dir
        self.lmdb_dir = lmdb_dir
        self.lmdb_mode = lmdb_mode
        self.metadata_source = metadata_source
        self.min_tag_score = min_tag_score
        self.stop_file = stop_file
        self.train_max_side = max(256, int(train_max_side or 1024))
        self.lmdb_env = None
        self.lmdb_ready = False
        self.lmdb_has_index_keys = False
        self.lmdb_has_path_keys = False

        augment_mode = str(augment_mode or "light").strip().lower()
        train_ops = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if augment_mode == "full":
            train_ops.append(transforms.RandAugment(num_ops=2, magnitude=6))
        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        if augment_mode in {"light", "full"}:
            train_ops.append(transforms.RandomErasing(p=0.10 if augment_mode == "light" else 0.15, scale=(0.02, 0.12)))
        self.transform_train = transforms.Compose(train_ops)
        self.transform_val = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.transform = self.transform_train
        self.is_train = True
        log.info(f"Train decode: max_side={self.train_max_side}, augment={augment_mode}")

        log.info(f"Scanning JSON/metadata under: {data_dir}")
        self.samples = self._scan(data_dir)
        log.info(f"Found {len(self.samples)} valid (image, text) pairs.")

        if not self.samples:
            return

        self.cached_embeddings = self._load_or_build_cache(text_model_name, cache_file, device)
        if lmdb_mode != "off":
            self._prepare_lmdb(rebuild=rebuild_lmdb)

    def _scan(self, data_dir: str) -> list[dict[str, str]]:
        samples: list[dict[str, str]] = []
        txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
        skipped = 0
        found = 0
        for m_path in txt_files:
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                parent = os.path.dirname(m_path)
                for line in lines:
                    if " - " in line:
                        fname, tags_str = line.strip().split(" - ", 1)
                        img_path = os.path.join(parent, fname.strip())
                        if not os.path.exists(img_path):
                            for ext in self.IMG_EXTS:
                                if os.path.exists(img_path + ext):
                                    img_path = img_path + ext
                                    break
                        if os.path.exists(img_path):
                            samples.append({"image_path": img_path, "description": tags_str.strip()})
                            found += 1
                        else:
                            skipped += 1
            except Exception as e:
                log.warning(f"Read error {m_path}: {e}")

        json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
        detailed_dirs = {
            os.path.dirname(path)
            for path in json_files
            if os.path.basename(path).lower() == "metadata_detailed.json"
        }
        log.info(f"Metadata files found: {len(json_files) + len(txt_files)}")

        for j_path in json_files:
            try:
                with open(j_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log.warning(f"Could not read {j_path}: {e}")
                continue

            # Flat Kohya-style metadata: {"filename.png": "tag1, tag2"}
            if (
                isinstance(data, dict)
                and "images" not in data
                and "total_images" not in data
                and data
                and all(isinstance(v, str) for v in data.values())
            ):
                if self.metadata_source in {"detailed", "detailed_florence"}:
                    continue
                if self.metadata_source == "auto" and os.path.dirname(j_path) in detailed_dirs:
                    log.info(f"Skipping {os.path.basename(j_path)}: metadata_detailed.json present in same folder")
                    continue
                parent = os.path.dirname(j_path)
                for fname, tags_str in data.items():
                    img_path = os.path.normpath(os.path.join(parent, fname))
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    samples.append({"image_path": img_path, "description": tags_str.strip()})
                    found += 1
                continue

            # Nested metadata: {"folder": ..., "total_images": N, "images": {fname: {...}}}
            if (
                isinstance(data, dict)
                and "images" in data
                and isinstance(data.get("images"), dict)
                and "total_images" in data
            ):
                if self.metadata_source == "simple":
                    continue
                parent = os.path.dirname(j_path)
                images_section = data["images"]

                for fname, info in images_section.items():
                    if not isinstance(info, dict):
                        skipped += 1
                        continue

                    tags: list[str] = []
                    for key in ("character_tags", "general_tags"):
                        tag_list = info.get(key, [])
                        if isinstance(tag_list, list):
                            tags.extend(str(tag) for tag in tag_list if tag)

                    scores = info.get("all_scores", {})
                    if isinstance(scores, dict):
                        def score_value(item):
                            try:
                                return float(item[1])
                            except Exception:
                                return 0.0

                        sorted_tags = sorted(scores.items(), key=score_value, reverse=True)
                        scored_tags = [str(t) for t, s in sorted_tags[:40] if score_value((t, s)) >= self.min_tag_score]
                        for tag in scored_tags:
                            if tag not in tags:
                                tags.append(tag)

                    if not tags:
                        caption = str(info.get("caption_florence2", "")).strip()
                        if caption and self.metadata_source == "detailed_florence":
                            tags = [caption]
                        else:
                            skipped += 1
                            continue

                    desc_parts = [", ".join(tags[:30])]
                    caption = str(info.get("caption_florence2", "")).strip()
                    if caption and self.metadata_source == "detailed_florence":
                        desc_parts.append(caption)
                    desc = ". ".join(part for part in desc_parts if part)

                    img_path = os.path.normpath(os.path.join(parent, fname))
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue

                    samples.append({"image_path": img_path, "description": desc})
                    found += 1
                continue

            # Legacy folder metadata with sample_files & description
            if (
                isinstance(data, dict)
                and "sample_files" in data
                and "description" in data
            ):
                parent = os.path.dirname(j_path)
                sample_files = data["sample_files"]
                descriptions = [d.strip() for d in data["description"].split(";") if d.strip()]
                folder_desc = data.get("description", "")

                all_imgs: list[str] = []
                for ext in self.IMG_EXTS:
                    all_imgs.extend(glob.glob(os.path.join(parent, f"*{ext}")))
                if not all_imgs:
                    all_imgs = [os.path.join(parent, fn) for fn in sample_files]

                caption_map: dict[str, str] = {}
                if len(descriptions) == len(sample_files):
                    for fn, cap in zip(sample_files, descriptions):
                        caption_map[fn] = cap
                elif descriptions:
                    for fn in sample_files:
                        caption_map[fn] = folder_desc

                for img_path in all_imgs:
                    img_path = os.path.normpath(img_path)
                    if not os.path.exists(img_path):
                        skipped += 1
                        continue
                    fname = os.path.basename(img_path)
                    desc = caption_map.get(fname, folder_desc)
                    if not desc:
                        skipped += 1
                        continue
                    samples.append({"image_path": img_path, "description": desc})
                    found += 1
                continue

            # Dict of record dicts or list of records
            records = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()) else [data]
            for record in records:
                if not isinstance(record, dict):
                    skipped += 1
                    continue

                img_path = self._extract_key(record, self.IMG_KEYS)
                description = self._extract_key(record, self.TXT_KEYS)
                if not img_path:
                    base = os.path.splitext(j_path)[0]
                    for ext in self.IMG_EXTS:
                        if os.path.exists(base + ext):
                            img_path = base + ext
                            break
                if not img_path or not description:
                    skipped += 1
                    continue

                if not os.path.isabs(img_path):
                    img_path = os.path.abspath(os.path.join(os.path.dirname(j_path), img_path))
                img_path = os.path.normpath(img_path)
                if not os.path.exists(img_path):
                    skipped += 1
                    continue
                samples.append({"image_path": img_path, "description": str(description)})
                found += 1

        log.info(f"Records added: {found}")
        if skipped:
            log.warning(f"Skipped (missing file or description): {skipped}")
        return samples

    @staticmethod
    def _extract_key(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for k in keys:
            if k in d and d[k]:
                return d[k]
        return None

    def _load_or_build_cache(self, model_name: str, cache_file: str, device: torch.device) -> dict[str, torch.Tensor]:
        all_descriptions = [s["description"] for s in self.samples]
        data_hash = hashlib.md5(json.dumps(sorted(set(all_descriptions)), ensure_ascii=False).encode()).hexdigest()

        if os.path.exists(cache_file):
            log.info(f"Found cache file: {cache_file}. Checking freshness...")
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=False)
                if saved.get("hash") == data_hash:
                    log.info("Cache is fresh. Loading...")
                    return saved["embeddings"]
                else:
                    log.info("Cache is stale. Recomputing...")
            except Exception as e:
                log.warning(f"Cache read error: {e}. Recomputing...")

        embeddings = self._build_cache(model_name, all_descriptions, device)
        torch.save({"hash": data_hash, "embeddings": embeddings}, cache_file)
        log.info(f"Cache saved: {cache_file}")
        return embeddings

    @staticmethod
    def _build_cache(model_name: str, descriptions: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        unique = list(set(descriptions))
        log.info(f"Computing embeddings for {len(unique)} unique descriptions...")
        embeddings: dict[str, torch.Tensor] = {}

        batch_size = 256
        if SentenceTransformer is not None:
            log.info(f"Loading SentenceTransformer ({model_name})...")
            text_model = SentenceTransformer(model_name, device=str(device))
            text_model.eval()
            for i in range(0, len(unique), batch_size):
                batch = unique[i : i + batch_size]
                with torch.no_grad():
                    vecs = text_model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                    for desc, vec in zip(batch, vecs):
                        embeddings[desc] = vec.cpu()
            del text_model
        else:
            log.warning("sentence-transformers not installed. Using AutoTokenizer/AutoModel fallback.")
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            text_model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
            fallback_batch = 64
            for i in range(0, len(unique), fallback_batch):
                batch = unique[i : i + fallback_batch]
                with torch.no_grad():
                    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    outputs = text_model(**inputs)
                    token_embeddings = outputs.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    summed = (token_embeddings * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-6)
                    vecs = F.normalize(summed / counts, p=2, dim=1)
                    for desc, vec in zip(batch, vecs):
                        embeddings[desc] = vec.cpu()
            del text_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return embeddings

    def __len__(self) -> int:
        return len(self.samples)

    def _open_training_image(self, sample: dict[str, str], image_bytes: bytes | None) -> Image.Image:
        if image_bytes is not None:
            raw_cm = Image.open(io.BytesIO(image_bytes))
        else:
            raw_cm = Image.open(sample["image_path"])
        with raw_cm as raw:
            try:
                raw.draft("RGB", (self.train_max_side, self.train_max_side))
            except Exception:
                pass
            image = raw.convert("RGB")
        if max(image.size) > self.train_max_side:
            image.thumbnail((self.train_max_side, self.train_max_side), Image.Resampling.BILINEAR)
        return image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        try:
            image_bytes = self._read_lmdb_image(idx) if self.lmdb_ready else None
            image = self._open_training_image(sample, image_bytes)
            img_tensor = self.transform(image)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            img_tensor = torch.zeros((3, 224, 224))

        text_emb = self.cached_embeddings[sample["description"]]
        return img_tensor, text_emb

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["lmdb_env"] = None
        return state

    def close(self) -> None:
        if self.lmdb_env is not None:
            self.lmdb_env.close()
            self.lmdb_env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _lmdb_hash(self) -> str:
        records = []
        for sample in self.samples:
            path = sample["image_path"]
            try:
                stat = os.stat(path)
                size = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                size = 0
                mtime = 0
            records.append((path, size, mtime, sample["description"]))
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def cancel_requested(self) -> bool:
        return bool(self.stop_file and os.path.exists(self.stop_file))

    def _raise_if_cancelled(self) -> None:
        if self.cancel_requested():
            raise KeyboardInterrupt("Training cancelled by UI.")

    def _prepare_lmdb(self, rebuild: bool = False) -> None:
        if lmdb is None:
            message = "LMDB unavailable: install the `lmdb` package."
            if self.lmdb_mode == "on":
                raise RuntimeError(message)
            log.warning(message + " Continuing without LMDB.")
            return
        if not self.lmdb_dir:
            return

        lmdb_path = Path(self.lmdb_dir)
        lmdb_path.mkdir(parents=True, exist_ok=True)
        expected_hash = self._lmdb_hash()
        meta_path = lmdb_path / "meta.json"
        if not rebuild:
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                    if meta.get("hash") == expected_hash and int(meta.get("count", 0)) == len(self.samples):
                        self.lmdb_ready = True
                        self.lmdb_has_index_keys = True
                        log.info(f"LMDB is up to date: {lmdb_path}")
                        return
                except Exception as exc:
                    log.warning(f"LMDB meta damaged: {exc}")

            if self._probe_existing_lmdb(lmdb_path):
                return

        total_bytes = 0
        for sample in self.samples:
            try:
                total_bytes += os.path.getsize(sample["image_path"])
            except OSError:
                pass
        map_size = max(128 << 20, int(total_bytes * DEFAULT_CONFIG["lmdb_map_headroom"]) + (64 << 20))
        log.info(f"Building LMDB: {lmdb_path} | samples={len(self.samples)} | map_size={format_size(map_size)}")
        env = lmdb.open(
            str(lmdb_path),
            map_size=map_size,
            subdir=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        written = skipped = 0
        txn = env.begin(write=True)
        try:
            for index, sample in enumerate(self.samples):
                if index % 50 == 0:
                    self._raise_if_cancelled()
                try:
                    with open(sample["image_path"], "rb") as handle:
                        payload = handle.read()
                    txn.put(f"img:{index}".encode("ascii"), payload)
                    try:
                        rel_path = clean_rel_path(Path(sample["image_path"]).resolve().relative_to(Path(self.data_dir).resolve()))
                        txn.put(path_key(rel_path), payload)
                    except Exception:
                        pass
                    written += 1
                except OSError:
                    skipped += 1
                if index > 0 and index % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            env.close()
            raise
        env.sync()
        env.close()

        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "hash": expected_hash,
                    "count": len(self.samples),
                    "written": written,
                    "skipped": skipped,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        self.lmdb_ready = True
        self.lmdb_has_index_keys = True
        self.lmdb_has_path_keys = True
        log.info(f"LMDB ready: written={written}, skipped={skipped}")

    def _probe_existing_lmdb(self, lmdb_path: Path) -> bool:
        data_file = lmdb_path / "data.mdb"
        if not data_file.exists():
            return False
        try:
            env = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
            with env.begin(write=False) as txn:
                indexed_ok = True
                path_ok = True
                for index, sample in enumerate(self.samples[: min(25, len(self.samples))]):
                    if txn.get(f"img:{index}".encode("ascii")) is None:
                        indexed_ok = False
                    try:
                        rel_path = clean_rel_path(Path(sample["image_path"]).resolve().relative_to(Path(self.data_dir).resolve()))
                    except Exception:
                        rel_path = clean_rel_path(Path(sample["image_path"]).name)
                    if txn.get(path_key(rel_path)) is None:
                        path_ok = False
                    if not indexed_ok and not path_ok:
                        break
            env.close()
            if indexed_ok or path_ok:
                self.lmdb_ready = True
                self.lmdb_has_index_keys = indexed_ok
                self.lmdb_has_path_keys = path_ok
                mode = "img:N" if indexed_ok else "path keys"
                log.info(f"LMDB found and opened read-only without rebuild: {lmdb_path} ({mode})")
                return True
        except Exception as exc:
            log.warning(f"Could not open existing LMDB read-only: {exc}")
        return False

    def _open_lmdb(self):
        if not self.lmdb_ready or lmdb is None or not self.lmdb_dir:
            return None
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_dir),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
        return self.lmdb_env

    def _read_lmdb_image(self, idx: int) -> bytes | None:
        env = self._open_lmdb()
        if env is None:
            return None
        with env.begin(write=False) as txn:
            value = txn.get(f"img:{idx}".encode("ascii")) if self.lmdb_has_index_keys else None
            if value is None and self.lmdb_has_path_keys:
                sample = self.samples[idx]
                try:
                    rel_path = clean_rel_path(Path(sample["image_path"]).resolve().relative_to(Path(self.data_dir).resolve()))
                except Exception:
                    rel_path = clean_rel_path(Path(sample["image_path"]).name)
                value = txn.get(path_key(rel_path))
        return bytes(value) if value is not None else None

    def set_train(self) -> None:
        self.is_train = True
        self.transform = self.transform_train

    def set_val(self) -> None:
        self.is_train = False
        self.transform = self.transform_val


# ─────────────────────────────────────────────────────────────────────────────
# Loss, EMA, Multi-Scale Extraction, and Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """0.7 * CosineEmbeddingLoss + 0.3 * MSELoss + L2 weight decay."""

    def __init__(self, cosine_w: float = 0.7, mse_w: float = 0.3, l2_lambda: float = 1e-3):
        super().__init__()
        self.cosine_w = cosine_w
        self.mse_w = mse_w
        self.l2_lambda = l2_lambda
        self.cosine = nn.CosineEmbeddingLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = F.normalize(target, p=2, dim=1)
        pred_norm = F.normalize(pred, p=2, dim=1)
        ones = torch.ones(pred.size(0), device=pred.device)
        cos_loss = self.cosine(pred_norm, target_norm, ones)
        mse_loss = self.mse(pred_norm, target_norm)
        l2_reg = torch.mean(pred.pow(2))
        return self.cosine_w * cos_loss + self.mse_w * mse_loss + self.l2_lambda * l2_reg


class EMA:
    """Exponential Moving Average of trainable parameters for checkpoint stability."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])


def extract_multiscale_dino(dino_model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract CLS token + mean spatial patch pooling = (B, 1536) multi-scale vector."""
    hs = dino_model(pixel_values=pixel_values).last_hidden_state
    cls_tok = hs[:, 0, :]
    spatial = hs[:, 1:, :].mean(dim=1)
    return torch.cat([cls_tok, spatial], dim=-1)


def save_checkpoint(
    path: str,
    epoch: int,
    adapter: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    best_metric: float,
    ema: EMA | None,
    cfg: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "adapter_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "ema_shadow": ema.shadow if ema else None,
            "best_metric": best_metric,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: str,
    adapter: nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    ema: EMA | None,
    device: torch.device,
) -> tuple[int, float, dict[str, Any]]:
    log.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    if ema and ckpt.get("ema_shadow"):
        ema.shadow = ckpt["ema_shadow"]
    best_metric = ckpt.get("best_metric", ckpt.get("best_loss", float("inf")))
    return ckpt["epoch"] + 1, best_metric, ckpt.get("config", {})


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def gpu_status_text(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "CPU"
    try:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        return f"VRAM torch {format_size(allocated, 1)}/{format_size(reserved, 1)} | total {format_size(total, 1)}"
    except Exception:
        return "CUDA"


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser & Auto-Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DINOv2 (multi-scale) adapter → text embeddings")
    p.add_argument("--data_dir", default=DEFAULT_CONFIG["data_dir"])
    p.add_argument("--base_weights_dir", default=DEFAULT_CONFIG["base_weights_dir"])
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--grad_accum", type=int, default=DEFAULT_CONFIG["grad_accum_steps"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    p.add_argument("--hidden_dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--dropout", type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--text_model", default=DEFAULT_CONFIG["text_model"])
    p.add_argument("--dino_path", default=DEFAULT_CONFIG["dino_path"])
    p.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    p.add_argument("--train_max_side", type=int, default=DEFAULT_CONFIG["train_max_side"])
    p.add_argument("--augment", choices=["off", "light", "full"], default=DEFAULT_CONFIG["augment_mode"])
    p.add_argument("--lmdb", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--lmdb_dir", default=DEFAULT_CONFIG["lmdb_dir"])
    p.add_argument("--rebuild_lmdb", action="store_true")
    p.add_argument("--metadata_source", choices=["auto", "simple", "detailed", "detailed_florence"], default="auto")
    p.add_argument("--min_tag_score", type=float, default=0.1)
    p.add_argument("--no_auto_optimize", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--manga", action="store_true")
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--stop_file", default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--no_tqdm", action="store_true")
    return p.parse_args(argv)


def auto_optimize_args(args: argparse.Namespace, device: torch.device, dataset_size: int) -> None:
    if args.no_auto_optimize:
        log.info("Auto-optimize disabled (--no_auto_optimize).")
        return

    cpu_count = os.cpu_count() or 1
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb < 6:
            target_batch, target_accum = 8, 4
        elif vram_gb < 10:
            target_batch, target_accum = 16, 3
        elif vram_gb < 16:
            target_batch, target_accum = 24, 2
        else:
            target_batch, target_accum = 32, 2
        target_workers = min(4, max(2, cpu_count // 2)) if os.name == "nt" else min(8, max(2, cpu_count - 2))
    else:
        target_batch, target_accum = 4, 8
        target_workers = 0 if os.name == "nt" else min(4, max(1, cpu_count - 1))

    if dataset_size < 512:
        target_batch = min(target_batch, 8)
    elif dataset_size > 20000 and device.type == "cuda" and os.name != "nt":
        target_workers = min(target_workers + 2, 12)

    args.batch_size = max(1, min(args.batch_size, target_batch))
    args.grad_accum = max(args.grad_accum, target_accum)
    args.num_workers = target_workers
    log.info(
        f"Auto-optimize: batch={args.batch_size}, accum={args.grad_accum}, "
        f"effective={args.batch_size * args.grad_accum}, workers={args.num_workers}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.cpu:
        device = torch.device("cpu")
        log.warning("CPU mode (--cpu). Training will be slow.")
    elif not torch.cuda.is_available():
        log.error("CUDA is not available! Install PyTorch with CUDA support or pass --cpu.")
        return 1
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"CUDA: {gpu_name} | VRAM: {format_size(vram * 1024**3, precision=1)}")
        torch.backends.cudnn.benchmark = True

    base_dir = args.base_weights_dir
    os.makedirs(base_dir, exist_ok=True)
    cache_file = os.path.join(base_dir, DEFAULT_CONFIG["cache_file"])

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Run artifacts will be saved under: {save_dir}")

    dataset = ImageTextDataset(
        data_dir=args.data_dir,
        text_model_name=args.text_model,
        cache_file=cache_file,
        device=device,
        lmdb_dir=args.lmdb_dir,
        lmdb_mode=args.lmdb,
        rebuild_lmdb=args.rebuild_lmdb,
        metadata_source=args.metadata_source,
        min_tag_score=args.min_tag_score,
        stop_file=args.stop_file,
        train_max_side=args.train_max_side,
        augment_mode=args.augment,
    )
    if len(dataset) == 0:
        log.error("No training data! Check metadata format and image paths.")
        return 1

    auto_optimize_args(args, device, len(dataset))
    sample_emb = next(iter(dataset.cached_embeddings.values()))
    output_dim = sample_emb.shape[0]
    log.info(f"Text embedding dimension (output_dim): {output_dim}")

    safe_workers = max(0, int(args.num_workers))
    if os.name == "nt" and not dataset.lmdb_ready:
        safe_workers = min(safe_workers, 2)
    if dataset.lmdb_ready:
        log.info(f"DataLoader will read images from LMDB: {args.lmdb_dir}")

    val_split = DEFAULT_CONFIG["val_split"]
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    all_indices = list(range(n_total))
    _rng.seed(42)
    _rng.shuffle(all_indices)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]
    log.info(f"Split: train={n_train}, val={n_val} ({val_split:.0%})")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=safe_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(safe_workers > 0),
        prefetch_factor=4 if safe_workers > 0 else None,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=safe_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(safe_workers > 0),
        prefetch_factor=4 if safe_workers > 0 else None,
    )

    log.info(f"Loading DINOv2 from: {args.dino_path}")
    dinov2 = AutoModel.from_pretrained(args.dino_path, local_files_only=True)
    dinov2.to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False
    total_params = sum(p.numel() for p in dinov2.parameters())
    log.info(f"DINOv2 loaded. Frozen parameters: {total_params:,}")

    from .models import DINOv2Adapter
    input_dim = DEFAULT_CONFIG["input_dim"]
    adapter = DINOv2Adapter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = CombinedLoss(
        cosine_w=DEFAULT_CONFIG["cosine_loss_weight"],
        mse_w=DEFAULT_CONFIG["mse_loss_weight"],
        l2_lambda=DEFAULT_CONFIG["l2_lambda"],
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = -(-len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs + 1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, total_steps),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ema = None if args.no_ema else EMA(adapter, decay=DEFAULT_CONFIG["ema_decay"])

    start_epoch = 0
    best_metric = float("inf")
    best_metric_filename = "manga_best_metric.json" if args.manga else "best_metric.json"
    best_metric_file = os.path.join(base_dir, best_metric_filename)
    dataset_hash = hashlib.md5(
        json.dumps([(s["image_path"], s["description"]) for s in sorted(dataset.samples, key=lambda x: x["image_path"])], ensure_ascii=False).encode()
    ).hexdigest()

    if os.path.exists(best_metric_file):
        try:
            with open(best_metric_file, "r") as f:
                saved = json.load(f)
            if saved.get("dataset_hash") == dataset_hash:
                best_metric = saved["best_metric"]
                log.info(f"Loaded previous best_metric={best_metric:.4f}")
        except Exception:
            pass

    if args.resume:
        if not os.path.exists(args.resume):
            log.error(f"Checkpoint not found: {args.resume}")
            return 1
        start_epoch, best_metric, _ = load_checkpoint(args.resume, adapter, optimizer, scheduler, scaler, ema, device)
        log.info(f"Resuming from epoch {start_epoch} | best_metric={best_metric:.4f}")
    else:
        target_adapter = "manga_adapter.pth" if args.manga else "best_adapter.pth"
        existing = sorted(glob.glob(os.path.join(base_dir, "**", target_adapter), recursive=True))
        if existing and not args.fresh:
            latest = existing[-1]
            if args.finetune:
                log.info(f"Auto-loading adapter: {latest} (--finetune)")
                answer = "y"
            else:
                answer = "n"
            if answer == "y":
                try:
                    start_epoch, best_metric, _ = load_checkpoint(latest, adapter, optimizer, scheduler, scaler, ema, device)
                except Exception:
                    ckpt = torch.load(latest, map_location=device, weights_only=False)
                    sd = ckpt.get("adapter", ckpt) if isinstance(ckpt, dict) else ckpt
                    adapter.load_state_dict(sd)
                    start_epoch = 0

    patience = DEFAULT_CONFIG["early_stopping_patience"]
    no_improve = 0
    grad_accum = args.grad_accum
    label_noise = DEFAULT_CONFIG["label_noise"]

    log.info("=" * 55)
    log.info(f"Training: epochs {start_epoch + 1} -> {args.epochs} | batch={args.batch_size} x {grad_accum}")
    log.info("=" * 55)

    try:
        for epoch in range(start_epoch, args.epochs):
            if dataset.cancel_requested():
                raise KeyboardInterrupt
            adapter.train()
            dataset.set_train()
            total_loss = 0.0
            n_batches = 0
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            for step, (images, text_embs) in enumerate(train_loader):
                if step % 10 == 0 and dataset.cancel_requested():
                    raise KeyboardInterrupt
                images = images.to(device, non_blocking=True)
                text_embs = text_embs.to(device, non_blocking=True)
                if label_noise > 0:
                    text_embs = text_embs + torch.randn_like(text_embs) * label_noise

                with torch.amp.autocast(device_type="cuda", enabled=(str(device) == "cuda")):
                    with torch.no_grad():
                        dino_features = extract_multiscale_dino(dinov2, images)
                    pred_embs = adapter(dino_features)
                    loss = criterion(pred_embs, text_embs) / grad_accum

                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(adapter)

                total_loss += loss.item() * grad_accum
                n_batches += 1
                current_loss = loss.item() * grad_accum
                current_lr = optimizer.param_groups[0]["lr"]

                log_every = max(1, int(args.log_every))
                if (step + 1) % log_every == 0 or (step + 1) == len(train_loader):
                    elapsed_step = time.time() - t0
                    done = step + 1
                    speed = done / max(elapsed_step, 1e-6)
                    eta = (len(train_loader) - done) / max(speed, 1e-6)
                    avg_so_far = total_loss / max(n_batches, 1)
                    log.info(
                        f"TRAIN epoch={epoch+1}/{args.epochs} step={done}/{len(train_loader)} "
                        f"loss={current_loss:.4f} avg_loss={avg_so_far:.4f} "
                        f"lr={current_lr:.2e} speed={speed:.2f} it/s ETA={format_duration(eta)} | "
                        f"{gpu_status_text(device)}"
                    )

            avg_train_loss = total_loss / max(n_batches, 1)
            elapsed = time.time() - t0

            adapter.eval()
            dataset.set_val()
            ema_backup = ema.apply(adapter) if ema else None
            val_loss = 0.0
            val_cos_sim = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for images, text_embs in val_loader:
                    if dataset.cancel_requested():
                        raise KeyboardInterrupt
                    images = images.to(device, non_blocking=True)
                    text_embs = text_embs.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", enabled=(str(device) == "cuda")):
                        dino_features = extract_multiscale_dino(dinov2, images)
                        pred_embs = adapter(dino_features)
                        loss = criterion(pred_embs, text_embs)
                        cos_sim = F.cosine_similarity(
                            F.normalize(pred_embs.float(), dim=1),
                            F.normalize(text_embs.float(), dim=1),
                            dim=1,
                        ).mean()

                    val_loss += loss.item()
                    val_cos_sim += cos_sim.item()
                    n_val_batches += 1

            if ema and ema_backup:
                ema.restore(adapter, ema_backup)

            avg_val_loss = val_loss / max(n_val_batches, 1)
            avg_cos_sim = val_cos_sim / max(n_val_batches, 1)
            current_metric = avg_val_loss - avg_cos_sim

            log.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
                f"val_cos_sim={avg_cos_sim:.4f} | metric={current_metric:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1:04d}.pth")
            save_checkpoint(ckpt_path, epoch, adapter, optimizer, scheduler, scaler, best_metric, ema, vars(args))

            if current_metric < best_metric:
                best_metric = current_metric
                no_improve = 0
                if ema:
                    ema_backup = ema.apply(adapter)
                    save_data = {"adapter": adapter.state_dict()}
                    ema.restore(adapter, ema_backup)
                else:
                    save_data = {"adapter": adapter.state_dict()}

                target_name = "manga_adapter.pth" if args.manga else "best_adapter.pth"
                best_path = os.path.join(save_dir, target_name)
                global_best_path = os.path.join(base_dir, target_name)
                torch.save(save_data, best_path)
                torch.save(save_data, global_best_path)

                with open(best_metric_file, "w") as f:
                    json.dump({"best_metric": best_metric, "dataset_hash": dataset_hash}, f)
                log.info(f"  * New best! val_loss={avg_val_loss:.4f}, cos_sim={avg_cos_sim:.4f} -> {best_path}")
            else:
                no_improve += 1
                log.info(f"  ! No improvement ({no_improve}/{patience})")
                if no_improve >= patience:
                    log.info(f"  [STOP] Early stopping: no improvement for {patience} epochs.")
                    break
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")

    if ema:
        ema.apply(adapter)
    dataset.close()

    target_name = "manga_adapter.pth" if args.manga else "best_adapter.pth"
    global_final_path = os.path.join(base_dir, target_name)
    torch.save({"adapter": adapter.state_dict()}, global_final_path)
    log.info(f"Training completed. Weights saved to: {global_final_path}")
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
