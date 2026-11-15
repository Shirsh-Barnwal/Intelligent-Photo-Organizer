from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .caching import ImageLmdbReader, ImagePathDataset, pil_collate, prepare_image_lmdb, resolve_lmdb_dir, scan_images
from .clustering import apply_folder_limit, cluster_images, deduplicate_images, group_sequences
from .models import ModelManager
from .search import run_ai_search
from .tagging import (
    build_semantic_vector,
    contains_year_token,
    display_tag,
    normalized_tag,
    optimize_cluster_folder_names,
    prepend_character_folder,
    sanitize_subfolder_path,
    strip_year_tokens,
)
from .utils import (
    CONTENT_NEG_ANCHOR,
    SETTINGS_BOOL_KEYS,
    SETTINGS_NUMBER_KEYS,
    SETTINGS_STRING_KEYS,
    STYLE_NEG_ANCHOR,
    emit_log,
    emit_progress,
    get_paths,
    parse_tags,
    read_json,
    safe_filename,
    write_json,
)

logger = logging.getLogger(__name__)
_SHARED_MANAGER: ModelManager | None = None


@dataclass
class FlorenceQueueRecord:
    path: Path
    prompt_tags: str
    image: Image.Image | None
    error: str | None = None


class FlorenceQueueDataset(Dataset):
    """Dataset for batch caption generation with Florence-2."""

    def __init__(
        self,
        queued: list[tuple[Path, str]],
        max_side: int = 768,
        lmdb_dir: str | Path | None = None,
        lmdb_source: str | Path | None = None,
    ):
        self.queued = [(Path(path), str(prompt_tags)) for path, prompt_tags in queued]
        self.max_side = max(256, int(max_side))
        self.lmdb_dir = Path(lmdb_dir) if lmdb_dir else None
        self.lmdb_source = Path(lmdb_source).expanduser().resolve() if lmdb_source else None
        self._lmdb_reader: ImageLmdbReader | None = None

    def _reader(self) -> ImageLmdbReader | None:
        if self.lmdb_dir is None or self.lmdb_source is None:
            return None
        if self._lmdb_reader is None:
            self._lmdb_reader = ImageLmdbReader(self.lmdb_dir, self.lmdb_source)
        return self._lmdb_reader

    def close(self) -> None:
        if self._lmdb_reader is not None:
            self._lmdb_reader.close()
            self._lmdb_reader = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.queued)

    def __getitem__(self, idx: int) -> FlorenceQueueRecord:
        path, prompt_tags = self.queued[idx]
        try:
            reader = self._reader()
            if reader is None:
                with Image.open(path) as raw:
                    image = raw.convert("RGB")
            else:
                image = reader.open_image(path)
            image.thumbnail((self.max_side, self.max_side), Image.Resampling.BILINEAR)
            return FlorenceQueueRecord(path=path, prompt_tags=prompt_tags, image=image)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return FlorenceQueueRecord(path=path, prompt_tags=prompt_tags, image=None, error=str(exc))


def florence_collate(batch: list[FlorenceQueueRecord]) -> list[FlorenceQueueRecord]:
    return batch


def get_model_manager() -> ModelManager:
    """Return the singleton instance of ModelManager."""
    global _SHARED_MANAGER
    if _SHARED_MANAGER is None:
        _SHARED_MANAGER = ModelManager()
    return _SHARED_MANAGER


def get_runtime_status(precision_choice: str = "auto") -> str:
    return get_model_manager().get_status_text(precision_choice)


def run_pipeline(
    config: dict[str, Any],
    progress_callback: Callable[[float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    runner = PipelineRunner(config, progress_callback=progress_callback, log_callback=log_callback)
    return runner.run()


def run_metadata_export(
    config: dict[str, Any],
    progress_callback: Callable[[float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    runner = PipelineRunner(config, progress_callback=progress_callback, log_callback=log_callback)
    return runner.run_metadata_export()


def get_run_history(limit: int = 30) -> list[dict[str, Any]]:
    """Retrieve sorting run history log entries."""
    history_file = get_paths().logs_dir / "run_history.json"
    if not history_file.exists():
        return []
    try:
        with history_file.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
    except Exception:
        return []
    if not isinstance(entries, list):
        return []
    cleaned = [entry for entry in entries if isinstance(entry, dict)]
    cleaned.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return cleaned[: max(1, int(limit))]


def undo_last_move_run(target_dir: str | Path, log_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Revert the previous file-move operation using target folder's .undo_last_move.json journal."""
    target = Path(target_dir).expanduser().resolve()
    journal_file = target / ".undo_last_move.json"
    if not journal_file.exists():
        return {"ok": False, "message": f"Undo journal not found: {journal_file}"}

    payload = read_json(journal_file, {}, log_callback)
    operations = payload.get("operations", [])
    if not isinstance(operations, list) or not operations:
        return {"ok": False, "message": "Undo journal is empty."}

    restored = skipped = errors = 0
    remaining: list[dict[str, str]] = []
    for operation in reversed(operations):
        src = Path(str(operation.get("src", ""))).expanduser()
        dst = Path(str(operation.get("dst", ""))).expanduser()
        if not dst.exists():
            skipped += 1
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            destination = src
            suffix_counter = 1
            while destination.exists():
                destination = src.with_name(f"{src.stem}_restored_{suffix_counter}{src.suffix}")
                suffix_counter += 1
            shutil.move(str(dst), str(destination))
            restored += 1
        except Exception as exc:
            errors += 1
            remaining.append({"src": str(src), "dst": str(dst)})
            emit_log(log_callback, f"Undo warning for {dst.name}: {exc}")

    if remaining:
        payload["operations"] = list(reversed(remaining))
        write_json(journal_file, payload, log_callback)
    else:
        archived = journal_file.with_name(f".undo_last_move_applied_{time.strftime('%Y%m%d_%H%M%S')}.json")
        try:
            shutil.move(str(journal_file), str(archived))
        except Exception:
            pass

    return {
        "ok": errors == 0,
        "message": f"Undo finished. Restored: {restored}, Skipped: {skipped}, Errors: {errors}",
        "restored": restored,
        "skipped": skipped,
        "errors": errors,
    }


class PipelineRunner:
    """Core image sorting and categorization pipeline runner."""

    def __init__(
        self,
        config: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = dict(config)
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.manager = get_model_manager()
        self.manager.set_callbacks(progress_callback, log_callback)
        self.paths = get_paths()
        self.cancel_event: Event = self.config.get("cancel_event") or Event()
        self.stop_after_batch_event: Event = self.config.get("stop_after_batch_event") or Event()
        self.scan_status_callback = self.config.get("scan_status_callback")
        self.eta_callback: Callable[[dict[str, Any]], None] | None = self.config.get("eta_callback")
        self._timings: dict[str, float] = {}
        self._eta_speed_ema: float | None = None
        self._eta_stage = ""
        self._last_eta_emit = 0.0
        self._eta_last_completed = 0
        self._eta_last_ts = 0.0
        self._eta_update_count = 0
        self._stop_after_batch_triggered = False
        self._folder_uncertainty: dict[str, dict[str, Any]] = {}
        self._move_journal: list[dict[str, str]] = []
        self._move_journal_target: Path | None = None
        self._run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._image_lmdb: ImageLmdbReader | None = None
        self._image_lmdb_dir: Path | None = None
        self._image_lmdb_source: Path | None = None

    def log(self, message: str) -> None:
        emit_log(self.log_callback, message)

    def prog(self, value: float) -> None:
        emit_progress(self.progress_callback, value)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def _is_soft_stop_requested(self) -> bool:
        return self.stop_after_batch_event.is_set()

    def _mark_soft_stop(self, reason: str | None = None) -> None:
        if not self._stop_after_batch_triggered:
            self._stop_after_batch_triggered = True
            if reason:
                self.log(reason)

    def _batch_size(self, kind: str, key: str, fallback: int) -> int:
        raw = int(self.config.get(key, 0) or 0)
        requested = raw if raw > 0 else fallback
        return self.manager.resolve_batch_size(kind, requested)

    @staticmethod
    def _chunks(items: list[Any], size: int) -> list[list[Any]]:
        size = max(1, int(size))
        return [items[index : index + size] for index in range(0, len(items), size)]

    def _close_image_lmdb(self) -> None:
        if self._image_lmdb is not None:
            self._image_lmdb.close()
            self._image_lmdb = None

    def _open_image_for_path(self, path: Path) -> Image.Image:
        if self._image_lmdb is not None:
            try:
                return self._image_lmdb.open_image(path)
            except Exception as exc:
                self.log(f"Warning: LMDB image read failed for {path.name}, falling back to file: {exc}")
        with Image.open(path) as raw:
            return raw.convert("RGB")

    def _format_eta(self, eta_seconds: float) -> str:
        if eta_seconds <= 0:
            return "0s"
        mins, secs = divmod(int(eta_seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def _emit_eta(
        self,
        *,
        stage: str,
        completed: int,
        total: int,
        start_ts: float,
        force: bool = False,
    ) -> None:
        if stage != self._eta_stage:
            self._eta_stage = stage
            self._eta_speed_ema = None
            self._last_eta_emit = 0.0
            self._eta_last_completed = 0
            self._eta_last_ts = 0.0
            self._eta_update_count = 0
        now = time.time()
        elapsed = max(now - start_ts, 1e-6)
        delta_completed = max(completed - self._eta_last_completed, 0)
        delta_time = max(now - self._eta_last_ts, 1e-6) if self._eta_last_ts > 0 else 0.0
        if delta_completed > 0 and delta_time > 0:
            inst_speed = delta_completed / delta_time
        else:
            inst_speed = completed / elapsed
        alpha = 0.2
        if self._eta_speed_ema is None:
            self._eta_speed_ema = inst_speed
        else:
            self._eta_speed_ema = alpha * inst_speed + (1.0 - alpha) * self._eta_speed_ema
        speed = max(self._eta_speed_ema, 1e-6)
        remaining = max(total - completed, 0)
        eta_seconds = remaining / speed

        should_emit = force or (now - self._last_eta_emit >= 1.0)
        if not should_emit:
            return
        self._last_eta_emit = now
        self._eta_last_completed = completed
        self._eta_last_ts = now
        self._eta_update_count += 1

        warmup_mode = self._eta_update_count <= 1 and completed < total
        if warmup_mode:
            eta_text = "warming up..."
            speed_for_ui = 0.0
        else:
            eta_text = self._format_eta(eta_seconds)
            speed_for_ui = speed
        if self.eta_callback is None:
            self.log(f"  [{stage}] {completed}/{total} | {speed_for_ui:.1f} i/s | ETA: {eta_text}")
        else:
            try:
                self.eta_callback(
                    {
                        "stage": stage,
                        "completed": completed,
                        "total": total,
                        "speed": speed_for_ui,
                        "eta_seconds": eta_seconds,
                        "eta_text": eta_text,
                        "gpu_text": self.manager.gpu_status_text(),
                    }
                )
            except Exception:
                pass

    def run(self) -> dict[str, Any]:
        result: dict[str, Any]
        try:
            result = self._run_inner()
        except Exception as exc:
            self.log(f"Critical error in processing thread: {exc}")
            self.log(traceback.format_exc())
            result = {"ok": False, "message": "Failed due to an unexpected error."}
        finally:
            self._close_image_lmdb()
        result = self._flush_move_journal(result)
        self._append_run_history(result)
        return result

    def run_metadata_export(self) -> dict[str, Any]:
        try:
            return self._run_metadata_export_inner()
        except Exception as exc:
            self.log(f"Critical error in metadata export: {exc}")
            self.log(traceback.format_exc())
            return {"ok": False, "message": "Metadata export failed due to an unexpected error."}
        finally:
            self._close_image_lmdb()

    def _run_metadata_export_inner(self) -> dict[str, Any]:
        source = Path(self.config["source_dir"]).expanduser().resolve()
        if not source.is_dir():
            return {"ok": False, "message": f"Source not found: {source}"}

        load_config = dict(self.config)
        load_config.update(
            {
                "sort_ai_human": False,
                "sort_content": False,
                "sort_style": False,
                "sort_grouping": False,
                "sort_dedup": False,
                "sort_tagger_filter": False,
                "character_aware_recursive": False,
                "gen_metadata": True,
            }
        )
        if not self.manager.load_for_config(load_config, self.progress_callback, self.log_callback):
            return {"ok": False, "message": "Model loading failed for metadata export."}

        self.log("Scanning images for metadata export...")
        self.prog(0)
        img_paths = scan_images(
            source,
            recursive=bool(self.config.get("recursive_scan", True)),
            blocked_subfolders=self.config.get("blocked_subfolders", ""),
            cache_status_callback=self.scan_status_callback,
            log_callback=self.log_callback,
        )
        total = len(img_paths)
        if not total:
            return {"ok": False, "message": "No images found."}

        self._image_lmdb = prepare_image_lmdb(
            img_paths,
            source,
            self.config.get("lmdb_cache_dir") or resolve_lmdb_dir(None),
            mode=str(self.config.get("lmdb_mode", "auto")),
            rebuild=bool(self.config.get("rebuild_lmdb", False)),
            log_callback=self.log_callback,
        )
        if self._image_lmdb is not None:
            self._image_lmdb_dir = resolve_lmdb_dir(self.config.get("lmdb_cache_dir"))
            self._image_lmdb_source = source

        batch_size = self._batch_size("main", "batch_size_var", 8)
        tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", batch_size)
        florence_batch_size = self._batch_size("florence", "florence_batch_size", self.manager.florence_batch_size)
        tags_per_img = max(1, int(self.config.get("meta_tags_per_image", 30)))
        include_florence = bool(self.config.get("use_florence")) and self.manager.florence_model is not None
        max_side = max(256, int(self.config.get("florence_max_side", 768)))
        skip_existing_captions = bool(self.config.get("metadata_skip_existing_captions", True))
        save_every = max(0, int(self.config.get("metadata_save_every", 500)))
        florence_per_folder = max(0, int(self.config.get("metadata_florence_per_folder", 0)))
        flat_path = source / "metadata.json"
        detailed_path = source / "metadata_detailed.json"
        flat_metadata: dict[str, str] = {}
        detailed_metadata: dict[str, Any] = {
            "folder": str(source),
            "total_images": total,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lmdb_dir": str(self._image_lmdb_dir or resolve_lmdb_dir(self.config.get("lmdb_cache_dir"))),
            "images": {},
        }
        if detailed_path.exists():
            try:
                existing_detailed = read_json(detailed_path, {}, self.log_callback)
                if isinstance(existing_detailed, dict) and isinstance(existing_detailed.get("images"), dict):
                    detailed_metadata["images"].update(existing_detailed["images"])
                    self.log(f"Loaded existing detailed metadata: {len(existing_detailed['images'])} images")
            except Exception as exc:
                self.log(f"Warning: could not load existing detailed metadata: {exc}")
        if flat_path.exists():
            existing_flat = read_json(flat_path, {}, self.log_callback)
            if isinstance(existing_flat, dict):
                flat_metadata.update({str(key): str(value) for key, value in existing_flat.items() if isinstance(value, str)})

        planned_florence: set[str] = set()
        if include_florence:
            planned_counts: dict[str, int] = {}
            for path in img_paths:
                try:
                    rel_path = path.resolve().relative_to(source).as_posix()
                except Exception:
                    rel_path = path.name
                existing_item = detailed_metadata["images"].get(rel_path, {})
                if (
                    skip_existing_captions
                    and isinstance(existing_item, dict)
                    and str(existing_item.get("caption_florence2", "")).strip()
                ):
                    continue
                folder_key = str(Path(rel_path).parent).replace("\\", "/")
                if folder_key == ".":
                    folder_key = ""
                if florence_per_folder > 0 and planned_counts.get(folder_key, 0) >= florence_per_folder:
                    continue
                planned_florence.add(rel_path)
                planned_counts[folder_key] = planned_counts.get(folder_key, 0) + 1

        def flush_metadata(reason: str = "") -> None:
            detailed_metadata["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            detailed_metadata["total_images"] = total
            with flat_path.open("w", encoding="utf-8") as handle:
                json.dump(flat_metadata, handle, ensure_ascii=False, indent=2)
            with detailed_path.open("w", encoding="utf-8") as handle:
                json.dump(detailed_metadata, handle, ensure_ascii=False, indent=2)
            if reason:
                self.log(f"Metadata checkpoint saved ({reason}): {len(flat_metadata)} images")

        processed = errors = captions_generated = captions_skipped = 0
        export_start = time.time()
        self.log(f"Generating source metadata for {total} images...")
        self.log(f"Metadata batches: main={batch_size}, tagger={tagger_batch_size}, Florence={florence_batch_size}")

        dataset = ImagePathDataset(img_paths, lmdb_dir=self._image_lmdb_dir, lmdb_source=self._image_lmdb_source)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if sys.platform == "win32" else min(4, max((os.cpu_count() or 1) - 1, 1)),
            collate_fn=pil_collate,
        )

        for batch_records in loader:
            if self.is_cancelled():
                return {"ok": False, "message": "Cancelled."}
            valid = [(record.path, record.image) for record in batch_records if record.image is not None]
            if not valid:
                processed += len(batch_records)
                self.prog(processed / total * 90)
                continue

            paths_v = [item[0] for item in valid]
            imgs_v = [item[1] for item in valid]
            batch_preds = self.manager.wd_tagger_infer_batch(imgs_v)
            florence_queue: list[tuple[Path, str]] = []

            for path, preds in zip(paths_v, batch_preds):
                try:
                    rel_posix = path.resolve().relative_to(source).as_posix()
                except Exception:
                    rel_posix = path.name

                gen_tags, char_tags, rating, all_scores = self._metadata_tag_lists(preds, tags_per_img)
                comb_tags = ", ".join(char_tags + gen_tags)
                flat_metadata[rel_posix] = comb_tags
                entry = {
                    "rating": rating,
                    "general_tags": gen_tags,
                    "character_tags": char_tags,
                    "all_scores": all_scores,
                }
                detailed_metadata["images"][rel_posix] = entry
                if rel_posix in planned_florence:
                    florence_queue.append((path, comb_tags))

            if florence_queue and include_florence:
                self._run_florence_metadata_queue(florence_queue, detailed_metadata["images"])
                captions_generated += len(florence_queue)

            processed += len(batch_records)
            self.prog(processed / total * 90)
            self._emit_eta(stage="Metadata Export", completed=processed, total=total, start_ts=export_start)
            if save_every > 0 and processed % save_every < batch_size:
                flush_metadata(f"checkpoint {processed}/{total}")

            if self._is_soft_stop_requested():
                self._mark_soft_stop("Stop-after-batch requested.")
                break

        flush_metadata("final")
        self.prog(100)
        self.manager.cleanup()
        elapsed = time.time() - export_start
        return {
            "ok": True,
            "message": f"Source metadata export completed in {elapsed:.1f}s. Total: {processed}, Florence captions: {captions_generated}",
            "processed": processed,
            "captions": captions_generated,
        }

    def _metadata_tag_lists(self, preds: dict[str, float], tags_per_img: int) -> tuple[list[str], list[str], dict[str, float], dict[str, float]]:
        rating = {tag: round(preds.get(tag, 0.0), 3) for tag in ["explicit", "questionable", "sensitive", "general"]}
        general_tags: list[str] = []
        character_tags: list[str] = []
        all_scores: dict[str, float] = {}

        content_tags = [
            (tag, score)
            for tag, score in preds.items()
            if self.manager.wd_tag_categories.get(tag, -1) not in (9, "rating") and score > 0.05
        ]
        content_tags.sort(key=lambda item: item[1], reverse=True)
        for tag, score in content_tags:
            clean_tag = normalized_tag(tag)
            category = self.manager.wd_tag_categories.get(tag, -1)
            all_scores[clean_tag] = round(float(score), 4)
            if category in (4, "character"):
                if len(character_tags) < tags_per_img and clean_tag not in character_tags:
                    character_tags.append(clean_tag)
            elif len(general_tags) < tags_per_img and clean_tag not in general_tags:
                general_tags.append(clean_tag)
        return general_tags[:tags_per_img], character_tags[:tags_per_img], rating, all_scores

    def _run_inner(self) -> dict[str, Any]:
        source = Path(self.config["source_dir"]).expanduser().resolve()
        target = Path(self.config["target_dir"]).expanduser().resolve()
        if not self.config.get("dry_run"):
            target.mkdir(parents=True, exist_ok=True)
        self._move_journal_target = target

        initial_load_config = dict(self.config)
        deferred_florence = bool(initial_load_config.get("use_florence"))
        if deferred_florence:
            initial_load_config["use_florence"] = False
            self.log("Deferring Florence-2 load until metadata stage.")

        if not self.manager.load_for_config(initial_load_config, self.progress_callback, self.log_callback):
            return {"ok": False, "message": "Model loading failed."}

        self._timings = {"start_total": time.time()}
        self.log("Scanning images...")
        scan_start = time.time()
        self.prog(0)
        img_paths = scan_images(
            source,
            recursive=bool(self.config.get("recursive_scan", True)),
            blocked_subfolders=self.config.get("blocked_subfolders", ""),
            cache_status_callback=self.scan_status_callback,
            log_callback=self.log_callback,
        )
        self._timings["scan"] = time.time() - scan_start

        total = len(img_paths)
        if not total:
            return {"ok": False, "message": "No images found."}

        requested_batch = max(1, int(self.config.get("batch_size_var", 8)))
        batch_size = self._batch_size("main", "batch_size_var", requested_batch)
        self.log(f"Found {total} images. Batch: {batch_size}")
        self._image_lmdb = prepare_image_lmdb(
            img_paths,
            source,
            self.config.get("lmdb_cache_dir") or resolve_lmdb_dir(None),
            mode=str(self.config.get("lmdb_mode", "auto")),
            rebuild=bool(self.config.get("rebuild_lmdb", False)),
            log_callback=self.log_callback,
        )
        if self._image_lmdb is not None:
            self._image_lmdb_dir = resolve_lmdb_dir(self.config.get("lmdb_cache_dir"))
            self._image_lmdb_source = source

        do_ai = bool(self.config.get("sort_ai_human"))
        do_cont = bool(self.config.get("sort_content"))
        do_style = bool(self.config.get("sort_style"))
        do_group = bool(self.config.get("sort_grouping"))
        do_dedup = bool(self.config.get("sort_dedup"))

        group_sens = max(0.01, min(0.99, float(self.config.get("group_threshold", 0.35))))
        dedup_thr = max(0.5, min(0.999, float(self.config.get("dedup_threshold", 0.88))))

        group_descs = [tag.strip() for tag in str(self.config.get("group_desc_var", "")).split(",") if tag.strip()]
        use_sem_group = do_group and bool(group_descs)
        use_dino_group = do_group and not group_descs
        need_dino_feats = do_dedup or use_dino_group

        cont_pos, cont_neg = parse_tags(self.config.get("content_tags", ""), self.log_callback) if do_cont else ([], [])
        sty_pos, sty_neg = parse_tags(self.config.get("style_tags", ""), self.log_callback) if do_style else ([], [])
        uncertainty_threshold = max(0.5, min(0.99, float(self.config.get("uncertainty_threshold", 0.7))))
        route_uncertain = bool(self.config.get("route_uncertain", True))

        with torch.inference_mode():
            self.log("Initializing models and feature cache...")
            init_start = time.time()
            cont_min_conf = float(self.config.get("content_min_conf", 0.05))
            sty_min_conf = float(self.config.get("style_min_conf", 0.05))

            if do_cont:
                cont_pos_embeds = self.manager.get_tag_embeddings(cont_pos)
                cont_neg_embeds = self.manager.get_tag_embeddings(cont_neg)
                cont_anc_embed = self.manager.get_tag_embeddings([self.config.get("content_neg_anchor", CONTENT_NEG_ANCHOR)], is_anchor=True)
            else:
                cont_pos_embeds = cont_neg_embeds = cont_anc_embed = torch.empty(0, device=self.manager.device)

            if do_style:
                sty_pos_embeds = self.manager.get_tag_embeddings(sty_pos)
                sty_neg_embeds = self.manager.get_tag_embeddings(sty_neg)
                sty_anc_embed = self.manager.get_tag_embeddings([self.config.get("style_neg_anchor", STYLE_NEG_ANCHOR)], is_anchor=True)
            else:
                sty_pos_embeds = sty_neg_embeds = sty_anc_embed = torch.empty(0, device=self.manager.device)

            if use_sem_group:
                group_descs_embeds = self.manager.get_tag_embeddings(group_descs)
                group_anc_embed = self.manager.get_tag_embeddings(["none of the above, other, miscellaneous"], is_anchor=True)
                all_group_embeds = torch.cat([group_descs_embeds, group_anc_embed], dim=0)
            else:
                all_group_embeds = None

            self._timings["init"] = time.time() - init_start
            feat_cache_file = self.paths.feat_cache_file
            try:
                feat_cache = torch.load(str(feat_cache_file), weights_only=True) if feat_cache_file.exists() else {}
            except Exception as exc:
                self.log(f"Warning: failed to load feature cache: {exc}")
                feat_cache = {}
            cache_dirty = False

        dataset = ImagePathDataset(img_paths, lmdb_dir=self._image_lmdb_dir, lmdb_source=self._image_lmdb_source)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if sys.platform == "win32" else min(4, max((os.cpu_count() or 1) - 1, 1)),
            pin_memory=(self.manager.device == "cuda"),
            collate_fn=pil_collate,
        )

        accepted: list[tuple[Path, str]] = []
        dino_paths: list[Path] = []
        dino_feats: list[torch.Tensor] = []
        dino_subs: list[str] = []
        wd_preds_cache: dict[Path, dict[str, float]] = {}
        errors = skipped = processed = 0
        proc_end = 80

        need_tagger = bool(
            (self.config.get("sort_grouping") and not str(self.config.get("group_desc_var", "")).strip())
            or self.config.get("gen_metadata")
            or self.config.get("sort_tagger_filter")
            or (self.config.get("character_aware_recursive") and self.config.get("recursive_scan"))
        )
        can_stream = not need_dino_feats and not need_tagger and not self.config.get("gen_metadata") and not self.config.get("group_sequences")
        do_tagger_filter = bool(self.config.get("sort_tagger_filter"))
        tagger_pos, tagger_neg = parse_tags(self.config.get("tagger_filter_tags", ""), self.log_callback) if do_tagger_filter else ([], [])
        tagger_threshold = float(self.config.get("camie_threshold", 0.05))
        tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", batch_size)

        process_start = time.time()
        with torch.inference_mode():
            for batch_records in loader:
                if self.is_cancelled():
                    return {"ok": False, "message": "Cancelled."}

                valid = [(record.path, record.image) for record in batch_records if record.image is not None]
                bad_records = [record for record in batch_records if record.image is None]
                if bad_records:
                    errors += len(bad_records)
                    for record in bad_records:
                        self.log(f"Warning: cannot open {record.path.name}: {record.error or 'error'}")
                if not valid:
                    processed += len(batch_records)
                    self.prog(processed / total * proc_end)
                    continue

                paths_v = [item[0] for item in valid]
                imgs_v = [item[1] for item in valid]
                count = len(imgs_v)
                ok = [True] * count
                subs = [""] * count

                if do_ai:
                    self._run_ai_human_batch(imgs_v, paths_v, subs, route_uncertain=route_uncertain, uncertainty_threshold=uncertainty_threshold)

                siglip_embs: list[torch.Tensor | None] = [None] * count
                if do_cont or do_style or use_sem_group or use_dino_group:
                    missing_siglip_indices: list[int] = []
                    for index in range(count):
                        if not ok[index]:
                            continue
                        try:
                            mtime = str(paths_v[index].stat().st_mtime)
                        except OSError:
                            mtime = "0"
                        cache_key = f"siglip_{paths_v[index]}_{mtime}"
                        if cache_key in feat_cache:
                            siglip_embs[index] = feat_cache[cache_key].to(self.manager.device, non_blocking=True)
                        else:
                            missing_siglip_indices.append(index)

                    if missing_siglip_indices:
                        try:
                            missing_imgs = [imgs_v[index] for index in missing_siglip_indices]
                            batch_embs = self.manager.siglip_embed_images_batch(missing_imgs)
                            for cache_index, embedding in zip(missing_siglip_indices, batch_embs):
                                siglip_embs[cache_index] = embedding.unsqueeze(0)
                                try:
                                    mtime = str(paths_v[cache_index].stat().st_mtime)
                                except OSError:
                                    mtime = "0"
                                feat_cache[f"siglip_{paths_v[cache_index]}_{mtime}"] = siglip_embs[cache_index].cpu()
                                cache_dirty = True
                        except Exception as exc:
                            self.log(f"Warning: SigLIP embedding error: {exc}")

                if do_cont and (cont_pos or cont_neg):
                    alive = [index for index in range(count) if ok[index] and siglip_embs[index] is not None]
                    if alive:
                        results = self.manager.siglip_filter(
                            [siglip_embs[index] for index in alive if siglip_embs[index] is not None],
                            cont_pos,
                            cont_pos_embeds,
                            cont_neg,
                            cont_neg_embeds,
                            cont_anc_embed,
                            cont_min_conf,
                        )
                        for result_index, item_index in enumerate(alive):
                            passed, tag, _margin = results[result_index]
                            if not passed:
                                ok[item_index] = False
                                skipped += 1
                            else:
                                subs[item_index] = str(Path(subs[item_index]) / tag) if subs[item_index] else tag

                if do_style and (sty_pos or sty_neg):
                    alive = [index for index in range(count) if ok[index] and siglip_embs[index] is not None]
                    if alive:
                        results = self.manager.siglip_filter(
                            [siglip_embs[index] for index in alive if siglip_embs[index] is not None],
                            sty_pos,
                            sty_pos_embeds,
                            sty_neg,
                            sty_neg_embeds,
                            sty_anc_embed,
                            sty_min_conf,
                        )
                        for result_index, item_index in enumerate(alive):
                            passed, tag, _margin = results[result_index]
                            if not passed:
                                ok[item_index] = False
                                skipped += 1
                            else:
                                subs[item_index] = str(Path(subs[item_index]) / tag) if subs[item_index] else tag

                if do_tagger_filter and self.manager.wd_tagger is not None and (tagger_pos or tagger_neg):
                    alive = [index for index in range(count) if ok[index]]
                    missing_wd = [index for index in alive if paths_v[index] not in wd_preds_cache]
                    if missing_wd:
                        for chunk in self._chunks(missing_wd, tagger_batch_size):
                            batch_preds = self.manager.wd_tagger_infer_batch([imgs_v[index] for index in chunk])
                            for item_index, preds in zip(chunk, batch_preds):
                                if preds:
                                    wd_preds_cache[paths_v[item_index]] = preds
                    for item_index in alive:
                        preds = wd_preds_cache.get(paths_v[item_index], {})
                        passed, _fail_tag = self.manager.tagger_filter(preds, tagger_pos, tagger_neg, tagger_threshold, image_name=paths_v[item_index])
                        if not passed:
                            ok[item_index] = False
                            skipped += 1

                if self.config.get("character_aware_recursive") and self.manager.wd_tagger is not None:
                    alive = [index for index in range(count) if ok[index]]
                    missing_wd = [index for index in alive if paths_v[index] not in wd_preds_cache]
                    if missing_wd:
                        for chunk in self._chunks(missing_wd, tagger_batch_size):
                            batch_preds = self.manager.wd_tagger_infer_batch([imgs_v[index] for index in chunk])
                            for item_index, preds in zip(chunk, batch_preds):
                                if preds:
                                    wd_preds_cache[paths_v[item_index]] = preds
                    for item_index in alive:
                        subs[item_index] = prepend_character_folder(
                            paths_v[item_index],
                            subs[item_index],
                            wd_preds_cache.get(paths_v[item_index], {}),
                            self.config,
                            self.manager.wd_tag_categories,
                        )

                if use_sem_group and all_group_embeds is not None:
                    alive = [index for index in range(count) if ok[index] and siglip_embs[index] is not None]
                    for item_index in alive:
                        best_index, _score = self.manager.siglip_classify(siglip_embs[item_index], all_group_embeds)
                        if best_index != -1:
                            group_name = safe_filename(group_descs[best_index])
                            subs[item_index] = str(Path(subs[item_index]) / group_name) if subs[item_index] else group_name

                if need_dino_feats:
                    cache_dirty = self._collect_dino_features(
                        paths_v,
                        imgs_v,
                        ok,
                        subs,
                        siglip_embs,
                        feat_cache,
                        wd_preds_cache,
                        dino_paths,
                        dino_feats,
                        dino_subs,
                        need_tagger=need_tagger,
                        use_dino_group=use_dino_group,
                    ) or cache_dirty
                else:
                    for item_index in range(count):
                        if ok[item_index]:
                            if can_stream:
                                sanitized_sub = sanitize_subfolder_path(subs[item_index])
                                destination = target / sanitized_sub if sanitized_sub else target
                                if not self._place(paths_v[item_index], destination):
                                    errors += 1
                                    continue
                            accepted.append((paths_v[item_index], subs[item_index]))

                processed += len(batch_records)
                self.prog(processed / total * proc_end)
                self._emit_eta(stage="Main pipeline", completed=processed, total=total, start_ts=process_start, force=(processed >= total))
                if self._is_soft_stop_requested():
                    self._mark_soft_stop("Stop-after-batch requested.")
                    break

        return self._finalize_results(
            accepted=accepted,
            dino_paths=dino_paths,
            dino_feats=dino_feats,
            dino_subs=dino_subs,
            wd_preds_cache=wd_preds_cache,
            total=total,
            processed=processed,
            skipped=skipped,
            errors=errors,
            target=target,
            group_sens=group_sens,
            dedup_thr=dedup_thr,
            need_tagger=need_tagger,
            use_dino_group=use_dino_group,
            cache_dirty=cache_dirty,
            feat_cache=feat_cache,
            can_stream=can_stream,
        )

    def _run_ai_human_batch(
        self,
        imgs_v: list[Any],
        paths_v: list[Path],
        subs: list[str],
        *,
        route_uncertain: bool,
        uncertainty_threshold: float,
    ) -> None:
        try:
            inputs = self.manager.ai_proc(images=imgs_v, return_tensors="pt")
            inputs = {key: value.to(self.manager.device, non_blocking=True) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            with torch.amp.autocast(device_type=self.manager.device, dtype=self.manager.amp_dtype, enabled=(self.manager.use_amp and self.manager.device != "cpu")):
                logits = self.manager.ai_model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            indices = logits.argmax(-1).cpu().numpy()

            if self.config.get("use_sdxl_vote") and self.manager.sdxl_model is not None:
                sdxl_inputs = self.manager.sdxl_proc(images=imgs_v, return_tensors="pt")
                sdxl_inputs = {key: value.to(self.manager.device, non_blocking=True) for key, value in sdxl_inputs.items() if isinstance(value, torch.Tensor)}
                with torch.amp.autocast(device_type=self.manager.device, dtype=self.manager.amp_dtype, enabled=(self.manager.use_amp and self.manager.device != "cpu")):
                    sdxl_logits = self.manager.sdxl_model(**sdxl_inputs).logits
                sdxl_probs = torch.softmax(sdxl_logits.float(), dim=-1).cpu().numpy()
                sdxl_indices = sdxl_logits.argmax(-1).cpu().numpy()

                for index in range(len(imgs_v)):
                    label_1 = str(self.manager.ai_model.config.id2label[indices[index]]).lower()
                    label_2_raw = str(self.manager.sdxl_model.config.id2label[sdxl_indices[index]]).lower()
                    label_2 = "ai" if label_2_raw in {"artificial", "ai"} else "hum"
                    conf_1 = float(probs[index][indices[index]])
                    conf_2 = float(sdxl_probs[index][sdxl_indices[index]])
                    disagree = label_1 != label_2
                    low_conf = conf_1 < uncertainty_threshold or conf_2 < uncertainty_threshold
                    if route_uncertain and (disagree or low_conf):
                        subs[index] = "Needs_Review/AI_Confidence"
                    else:
                        subs[index] = label_1
            else:
                for index in range(len(imgs_v)):
                    label = str(self.manager.ai_model.config.id2label[indices[index]]).lower()
                    confidence = float(probs[index][indices[index]])
                    if route_uncertain and confidence < uncertainty_threshold:
                        subs[index] = "Needs_Review/AI_Confidence"
                    else:
                        subs[index] = label
        except Exception as exc:
            self.log(f"Warning: AI/Human error: {exc}")

    def _collect_dino_features(
        self,
        paths_v: list[Path],
        imgs_v: list[Any],
        ok: list[bool],
        subs: list[str],
        siglip_embs: list[torch.Tensor | None],
        feat_cache: dict[str, torch.Tensor],
        wd_preds_cache: dict[Path, dict[str, float]],
        dino_paths: list[Path],
        dino_feats: list[torch.Tensor],
        dino_subs: list[str],
        *,
        need_tagger: bool,
        use_dino_group: bool,
    ) -> bool:
        cache_dirty = False
        count = len(imgs_v)
        missing_dino_indices: list[int] = []
        dino_vecs: list[torch.Tensor | None] = [None] * count

        for index in range(count):
            if not ok[index]:
                continue
            try:
                mtime = str(paths_v[index].stat().st_mtime)
            except OSError:
                mtime = "0"
            cache_key = f"dinov2ms_{paths_v[index]}_{mtime}_{bool(self.config.get('use_dino_adapter', True))}"
            if cache_key in feat_cache:
                dino_vecs[index] = feat_cache[cache_key]
            else:
                missing_dino_indices.append(index)

        if missing_dino_indices:
            dino_batch_size = self._batch_size("dino", "dino_batch_size", len(missing_dino_indices))
            try:
                for chunk in self._chunks(missing_dino_indices, dino_batch_size):
                    dino_inputs = self.manager.dino_proc(images=[imgs_v[index] for index in chunk], return_tensors="pt")
                    dino_inputs = {key: value.to(self.manager.device, non_blocking=True) for key, value in dino_inputs.items() if isinstance(value, torch.Tensor)}
                    with torch.amp.autocast(device_type=self.manager.device, dtype=self.manager.amp_dtype, enabled=(self.manager.use_amp and self.manager.device != "cpu")):
                        hidden_state = self.manager.dino_model(**dino_inputs).last_hidden_state
                        cls_tok = hidden_state[:, 0, :]
                        spatial = hidden_state[:, 1:, :].mean(dim=1)
                        batch_feat = torch.cat([cls_tok, spatial], dim=-1)
                        if self.manager.dino_adapter is not None:
                            batch_feat = self.manager.dino_adapter(batch_feat)
                    batch_feat_cpu = batch_feat.float().cpu()
                    for batch_index, item_index in enumerate(chunk):
                        value = batch_feat_cpu[batch_index].unsqueeze(0)
                        dino_vecs[item_index] = value
                        try:
                            mtime = str(paths_v[item_index].stat().st_mtime)
                        except OSError:
                            mtime = "0"
                        feat_cache[f"dinov2ms_{paths_v[item_index]}_{mtime}_{bool(self.config.get('use_dino_adapter', True))}"] = value
                        cache_dirty = True
            except Exception as exc:
                self.log(f"Warning: DINOv2 extraction error: {exc}")

        alive_tagger = [index for index in range(count) if ok[index]]
        if need_tagger and self.manager.wd_tagger is not None and alive_tagger:
            missing_wd = [index for index in alive_tagger if paths_v[index] not in wd_preds_cache]
            if missing_wd:
                tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", len(missing_wd))
                for chunk in self._chunks(missing_wd, tagger_batch_size):
                    batch_preds = self.manager.wd_tagger_infer_batch([imgs_v[index] for index in chunk])
                    for item_index, preds in zip(chunk, batch_preds):
                        if preds:
                            wd_preds_cache[paths_v[item_index]] = preds

        for index in range(count):
            if not ok[index] or dino_vecs[index] is None:
                continue
            try:
                dino_vec = dino_vecs[index]
                sem_vec = None
                if need_tagger and self.manager.wd_tagger is not None:
                    preds = wd_preds_cache.get(paths_v[index])
                    if preds:
                        sem_vec = build_semantic_vector(preds, self.manager.wd_tag_categories, self.log_callback)

                if use_dino_group:
                    sig_vec = siglip_embs[index].cpu() if siglip_embs[index] is not None else torch.zeros_like(dino_vec)
                    dino_n = F.normalize(dino_vec, p=2, dim=-1) * 1.0
                    sig_n = F.normalize(sig_vec, p=2, dim=-1) * 0.7
                    parts = [dino_n, sig_n]
                    if sem_vec is not None:
                        sem_n = F.normalize(sem_vec, p=2, dim=-1) * 0.3
                        parts.append(sem_n)
                    combined = F.normalize(torch.cat(parts, dim=-1), p=2, dim=-1)
                    dino_feats.append(combined)
                else:
                    dino_feats.append(dino_vec)

                dino_paths.append(paths_v[index])
                dino_subs.append(subs[index])
            except Exception as exc:
                self.log(f"Warning: Feature aggregation error for {paths_v[index].name}: {exc}")

        return cache_dirty

    def _finalize_results(
        self,
        *,
        accepted: list[tuple[Path, str]],
        dino_paths: list[Path],
        dino_feats: list[torch.Tensor],
        dino_subs: list[str],
        wd_preds_cache: dict[Path, dict[str, float]],
        total: int,
        processed: int,
        skipped: int,
        errors: int,
        target: Path,
        group_sens: float,
        dedup_thr: float,
        need_tagger: bool,
        use_dino_group: bool,
        cache_dirty: bool,
        feat_cache: dict[str, torch.Tensor],
        can_stream: bool,
    ) -> dict[str, Any]:
        if self.is_cancelled():
            return {"ok": False, "message": "Cancelled."}

        if self.config.get("group_sequences") and not self.is_cancelled():
            group_sequences(accepted, dino_paths, dino_feats, dino_subs, self.log_callback, self.progress_callback)

        if self.config.get("sort_dedup") and dino_feats and not self.is_cancelled():
            before = len(dino_feats)
            self.log(f"Deduplicating {before} images...")
            dino_paths, dino_feats, dino_subs = deduplicate_images(dino_paths, dino_feats, dino_subs, dedup_thr)
            self.log(f"  Removed {before - len(dino_feats)} duplicates")
            self.prog(85)

        clustered_groups: dict[int, list[tuple[Path, str]]] = {}
        if use_dino_group and dino_feats and not self.is_cancelled():
            self.log(f"Clustering {len(dino_feats)} images...")
            try:
                if self.config.get("character_aware_recursive"):
                    buckets: dict[str, list[int]] = {}
                    for index, sub in enumerate(dino_subs):
                        parts = Path(sanitize_subfolder_path(sub)).parts if sub else ()
                        character_key = parts[0] if parts else ""
                        buckets.setdefault(character_key, []).append(index)
                    next_group_id = 0
                    for character_key, indices in buckets.items():
                        if not indices:
                            continue
                        self.log(f"  Character bucket '{character_key or 'root'}': {len(indices)} images")
                        bucket_groups = cluster_images(
                            [dino_paths[index] for index in indices],
                            [dino_feats[index] for index in indices],
                            [dino_subs[index] for index in indices],
                            group_sens,
                            max_clusters=int(self.config.get("max_folders_created", 0) or 0),
                            log_callback=self.log_callback,
                        )
                        for _bucket_id, members in bucket_groups.items():
                            clustered_groups[next_group_id] = members
                            next_group_id += 1
                else:
                    clustered_groups = cluster_images(
                        dino_paths,
                        dino_feats,
                        dino_subs,
                        group_sens,
                        max_clusters=int(self.config.get("max_folders_created", 0) or 0),
                        log_callback=self.log_callback,
                    )
            except Exception as exc:
                self.log(f"Warning: clustering error: {exc}")
                accepted.extend(list(zip(dino_paths, dino_subs)))
        elif dino_feats:
            accepted.extend(list(zip(dino_paths, dino_subs)))
        self.prog(85)

        uncat_indices = [index for index, (_path, sub) in enumerate(accepted) if not sub]
        if need_tagger and uncat_indices and not self.is_cancelled():
            self.log(f"Categorizing {len(uncat_indices)} 'Other' images using cached tags...")
            generic_tags = {"monochrome", "greyscale", "no_humans", "simple_background", "white_background", "sketch"}
            for item_index in uncat_indices:
                path, _sub = accepted[item_index]
                preds = wd_preds_cache.get(path)
                if preds:
                    filtered_tags = []
                    for tag, score in preds.items():
                        if contains_year_token(tag):
                            continue
                        if self.manager.wd_tag_categories.get(tag, -1) not in (9, "rating") and score > 0.35:
                            filtered_tags.append((tag, score * (0.1 if tag in generic_tags else 1.0)))
                    top_tags = sorted(filtered_tags, key=lambda item: item[1], reverse=True)[:2]
                    folder_name = safe_filename(strip_year_tokens(" ".join(display_tag(tag) for tag, _ in top_tags)))[:40] if top_tags else "Other"
                else:
                    folder_name = "Other"
                accepted[item_index] = (path, folder_name)

        if need_tagger and clustered_groups and not self.is_cancelled():
            self.log(f"Naming {len(clustered_groups)} clusters from tags...")
            accepted.extend(self._wd_tagger_name_clusters(clustered_groups, wd_preds_cache))
            self.prog(92)

        accepted = apply_folder_limit(
            accepted,
            int(self.config.get("max_folders_created", 0) or 0),
            self.paths.feat_cache_file,
            bool(self.config.get("use_dino_adapter", True)),
            sanitize_subfolder_path,
            self.log_callback,
        )

        do_meta = need_tagger and self.config.get("gen_metadata") and not self.is_cancelled() and not self._stop_after_batch_triggered and not self.config.get("dry_run")
        if do_meta:
            if self.config.get("use_florence"):
                self.log("Loading Florence-2 on demand for metadata...")
                if self.manager.device == "cuda" and bool(self.config.get("unload_inactive_models", True)):
                    self.manager.offload_non_florence_to_cpu()
                self.manager.load_for_config(self.config, self.progress_callback, self.log_callback)
            self.log(f"Generating rich metadata for {len(accepted)} images...")
            self._wd_tagger_generate_metadata(accepted, target, wd_preds_cache)

        if need_tagger:
            self.manager.unload_tagger()
            wd_preds_cache.clear()

        if cache_dirty and not self.is_cancelled():
            try:
                self.log("Saving feature cache...")
                feat_cache_path = self.paths.cache_dir / "feat_cache.pt"
                torch.save(feat_cache, str(feat_cache_path))
            except Exception as exc:
                self.log(f"Warning: could not save cache: {exc}")

        self.prog(95)
        placement_errors = 0
        if not can_stream:
            self.log(f"Saving {len(accepted)} images...")
            for index, (path, sub) in enumerate(accepted):
                sanitized_sub = sanitize_subfolder_path(sub)
                destination = target / sanitized_sub if sanitized_sub else target
                if not self._place(path, destination):
                    placement_errors += 1
                self.prog(95 + (index + 1) / max(len(accepted), 1) * 5)

        uncertain_saved = sum(1 for _path, sub in accepted if str(sub).startswith("Needs_Review"))
        total_time = time.time() - self._timings.get("start_total", time.time())
        self._timings["total"] = total_time
        self._save_timing_summary(processed)
        self.manager.cleanup()

        return {
            "ok": True,
            "message": f"Done in {total_time:.1f}s! Total: {total} | Processed: {processed} | Saved: {len(accepted) - placement_errors} | Filtered: {skipped} | Errors: {errors + placement_errors}",
            "processed": processed,
            "saved": len(accepted) - placement_errors,
            "uncertain_saved": uncertain_saved,
            "filtered": skipped,
            "errors": errors + placement_errors,
            "elapsed": total_time,
            "stopped_after_batch": self._stop_after_batch_triggered,
        }

    def _wd_tagger_name_clusters(
        self,
        clustered_groups: dict[int, list[tuple[Path, str]]],
        wd_preds_cache: dict[Path, dict[str, float]],
    ) -> list[tuple[Path, str]]:
        safe_naming_mode = bool(self.config.get("safe_naming_mode", True))
        consensus_ratio = float(self.config.get("name_consensus_ratio", 0.65))
        min_tag_score = float(self.config.get("name_min_tag_score", 0.18))
        char_min_score = float(self.config.get("name_char_min_score", 0.55))
        char_margin = float(self.config.get("name_char_margin", 0.12))
        max_parts = 3 if safe_naming_mode else 5

        exclude_tags = {"monochrome", "greyscale", "no_humans", "simple_background", "white_background", "sketch", "lineart", "highres", "absurdres"}
        weak_name_tags = {"1girl", "1boy", "multiple_girls", "multiple_boys", "solo", "portrait", "close_up"}

        cluster_names: dict[int, str] = {}
        cluster_candidate_tags: dict[int, list[tuple[str, float, float]]] = {}
        cluster_uncertain: dict[int, bool] = {}
        cluster_uncertain_reasons: dict[int, list[str]] = {}

        for cluster_id, members in clustered_groups.items():
            missing_paths = [path for path, _sub in members if path not in wd_preds_cache]
            if missing_paths and self.manager.wd_tagger is not None:
                batch_images = [self._open_image_for_path(p) for p in missing_paths]
                tagger_batch_size = self._batch_size("tagger", "tagger_batch_size", len(batch_images))
                for start in range(0, len(batch_images), tagger_batch_size):
                    batch_preds = self.manager.wd_tagger_infer_batch(batch_images[start : start + tagger_batch_size])
                    for p, preds in zip(missing_paths[start : start + tagger_batch_size], batch_preds):
                        wd_preds_cache[p] = preds

            total_images = max(len(members), 1)
            tag_count: dict[str, int] = {}
            tag_sum: dict[str, float] = {}
            char_count: dict[str, int] = {}
            char_sum: dict[str, float] = {}

            for path, _sub in members:
                preds = wd_preds_cache.get(path)
                if not preds:
                    continue
                seen_tag: set[str] = set()
                seen_char: set[str] = set()
                for raw_tag, score in preds.items():
                    norm_tag = normalized_tag(raw_tag)
                    category = self.manager.wd_tag_categories.get(raw_tag, -1)
                    if category in (9, "rating") or norm_tag in exclude_tags:
                        continue
                    if category in (4, "character"):
                        if score >= 0.2 and norm_tag not in seen_char:
                            char_count[norm_tag] = char_count.get(norm_tag, 0) + 1
                            seen_char.add(norm_tag)
                        if score >= 0.05:
                            char_sum[norm_tag] = char_sum.get(norm_tag, 0.0) + score
                        continue
                    if score < min_tag_score:
                        continue
                    if norm_tag not in seen_tag:
                        tag_count[norm_tag] = tag_count.get(norm_tag, 0) + 1
                        seen_tag.add(norm_tag)
                    tag_sum[norm_tag] = tag_sum.get(norm_tag, 0.0) + score

            consensus_tags = [(tag, count / total_images, tag_sum.get(tag, 0.0) / max(count, 1)) for tag, count in tag_count.items() if (count / total_images) >= consensus_ratio]
            consensus_tags.sort(key=lambda item: (item[1], item[2]), reverse=True)

            char_candidates = [(tag, count / total_images, char_sum.get(tag, 0.0) / max(count, 1)) for tag, count in char_count.items()]
            char_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)

            uncertain_reasons: list[str] = []
            selected_char: str | None = None
            if char_candidates:
                top_tag, top_cov, top_avg = char_candidates[0]
                second_avg = char_candidates[1][2] if len(char_candidates) > 1 else 0.0
                if top_avg >= char_min_score and (top_avg - second_avg) >= char_margin and top_cov >= max(0.4, consensus_ratio - 0.15):
                    selected_char = top_tag
                else:
                    uncertain_reasons.append("character_conflict_or_low_confidence")

            if not consensus_tags:
                uncertain_reasons.append("weak_tag_consensus")

            candidate_tags = list(consensus_tags)
            if selected_char is not None:
                candidate_tags.insert(0, (selected_char, 1.0, char_candidates[0][2] if char_candidates else 1.0))
            cluster_candidate_tags[cluster_id] = candidate_tags

            name_parts: list[str] = []
            if selected_char is not None:
                name_parts.append(display_tag(selected_char))
            for tag, _cov, _avg in consensus_tags:
                if len(name_parts) >= max_parts:
                    break
                if tag == selected_char or tag in weak_name_tags or contains_year_token(tag):
                    continue
                disp = display_tag(tag)
                if disp not in name_parts:
                    name_parts.append(disp)

            folder_name = safe_filename(strip_year_tokens(" ".join(name_parts)))[:60] if name_parts else f"group_{cluster_id + 1:03d}"
            cluster_names[cluster_id] = folder_name
            cluster_uncertain[cluster_id] = bool(uncertain_reasons)
            cluster_uncertain_reasons[cluster_id] = sorted(set(uncertain_reasons))

        final_names = optimize_cluster_folder_names(
            cluster_names,
            cluster_candidate_tags,
            self.config,
            siglip_embed_fn=(self.manager.siglip_embed_texts if self.manager.siglip_model else None),
            log_callback=self.log_callback,
        )

        if self.config.get("remember_cluster_names", True):
            memory = read_json(self.paths.cache_dir / "cluster_name_memory.json", {}, self.log_callback)
            if isinstance(memory, dict) and memory:
                for cluster_id, folder in list(final_names.items()):
                    remembered = memory.get(folder) or memory.get(sanitize_subfolder_path(folder))
                    if remembered:
                        final_names[cluster_id] = safe_filename(str(remembered))

        result: list[tuple[Path, str]] = []
        for cluster_id, members in clustered_groups.items():
            folder = final_names.get(cluster_id, f"group_{cluster_id + 1:03d}")
            for path, sub in members:
                parts = [part for part in [sanitize_subfolder_path(sub), sanitize_subfolder_path(folder)] if part]
                combined = str(Path(*parts)) if parts else ""
                result.append((path, combined))
                if combined:
                    self._folder_uncertainty[combined] = {
                        "uncertain": cluster_uncertain.get(cluster_id, False),
                        "uncertain_reasons": cluster_uncertain_reasons.get(cluster_id, []),
                    }
        return result

    def _run_florence_metadata_queue(self, caption_queue: list[tuple[Path, str]], images_meta: dict[str, Any]) -> None:
        if not caption_queue or self.manager.florence_model is None:
            return
        default_batch = 4
        batch_size = self._batch_size("florence", "florence_batch_size", default_batch)
        loader = DataLoader(
            FlorenceQueueDataset(caption_queue, max_side=768, lmdb_dir=self._image_lmdb_dir, lmdb_source=self._image_lmdb_source),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=florence_collate,
        )
        for records in loader:
            if self.is_cancelled():
                break
            valid = [r for r in records if r.image is not None]
            if valid:
                captions = self.manager.florence_infer_batch([r.image for r in valid if r.image is not None], [r.prompt_tags for r in valid])
                for record, caption in zip(valid, captions):
                    if caption:
                        images_meta.setdefault(record.path.name, {})["caption_florence2"] = caption
            if self._is_soft_stop_requested():
                break

    def _wd_tagger_generate_metadata(self, accepted: list[tuple[Path, str]], target: Path, wd_preds_cache: dict[Path, dict[str, float]]) -> None:
        folder_map: dict[str, list[Path]] = {}
        for path, sub in accepted:
            folder_map.setdefault(sub or "", []).append(path)

        for subfolder, paths in folder_map.items():
            if self.is_cancelled() or self._is_soft_stop_requested():
                break
            max_n = int(self.config.get("meta_max_per_folder", 0))
            sample = paths[:] if max_n <= 0 else random.sample(paths, min(max_n, len(paths)))
            tags_per_img = int(self.config.get("meta_tags_per_image", 30))
            images_meta: dict[str, Any] = {}
            florence_queue: list[tuple[Path, str]] = []

            for path in sample:
                preds = wd_preds_cache.get(path, {})
                if not preds:
                    images_meta[path.name] = {"error": "no tags found"}
                    continue
                gen_tags, char_tags, rating, all_scores = self._metadata_tag_lists(preds, tags_per_img)
                images_meta[path.name] = {
                    "rating": rating,
                    "general_tags": gen_tags,
                    "character_tags": char_tags,
                    "all_scores": all_scores,
                }
                if self.config.get("use_florence") and self.manager.florence_model is not None:
                    florence_queue.append((path, ", ".join(char_tags + gen_tags[:10])))

            if florence_queue:
                self._run_florence_metadata_queue(florence_queue, images_meta)

            metadata = {
                "folder": sanitize_subfolder_path(subfolder) or "root",
                "total_images": len(paths),
                "images": images_meta,
            }
            dst_dir = target / sanitize_subfolder_path(subfolder) if subfolder else target
            dst_dir.mkdir(parents=True, exist_ok=True)
            with (dst_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def _save_timing_summary(self, processed_count: int) -> None:
        try:
            self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.paths.logs_dir / f"sort_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
            with log_file.open("w", encoding="utf-8") as handle:
                handle.write("=== Image Sorter Performance Log ===\n")
                handle.write(f"Processed: {processed_count} images in {self._timings.get('total', 0):.2f}s\n")
        except Exception:
            pass

    def _flush_move_journal(self, result: dict[str, Any]) -> dict[str, Any]:
        if not self.config.get("move_files") or not self._move_journal:
            result["undo_available"] = False
            return result
        target = self._move_journal_target or Path(self.config.get("target_dir", ".")).expanduser().resolve()
        journal_file = target / ".undo_last_move.json"
        write_json(journal_file, {"operations": self._move_journal}, self.log_callback)
        result["undo_available"] = True
        return result

    def _append_run_history(self, result: dict[str, Any]) -> None:
        try:
            history_file = self.paths.logs_dir / "run_history.json"
            existing = read_json(history_file, [], self.log_callback)
            if not isinstance(existing, list):
                existing = []
            existing.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ok": bool(result.get("ok")),
                "message": str(result.get("message", "")),
                "processed": int(result.get("processed", 0)),
                "saved": int(result.get("saved", 0)),
            })
            write_json(history_file, existing[-200:], self.log_callback)
        except Exception:
            pass

    def _place(self, src: Path, dst_dir: Path) -> bool:
        destination = dst_dir / src.name
        if src.resolve() == destination.resolve():
            return True
        counter = 1
        while destination.exists():
            destination = dst_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            if self.config.get("dry_run"):
                return True
            dst_dir.mkdir(parents=True, exist_ok=True)
            mover = shutil.move if self.config.get("move_files") else shutil.copy2
            mover(src, destination)
            if self.config.get("move_files"):
                self._move_journal.append({"src": str(src), "dst": str(destination)})
            return True
        except Exception as exc:
            self.log(f"Warning: could not save {src}: {exc}")
            return False
