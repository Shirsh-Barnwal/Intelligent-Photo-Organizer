from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .utils import (
    VALID_EXTS,
    emit_log,
    format_size,
    get_paths,
    read_json,
    safe_filename,
    write_json,
)

try:
    import lmdb
except Exception:
    lmdb = None

DEFAULT_LMDB_NAME = "dinov2_train_lmdb"


# ─────────────────────────────────────────────────────────────────────────────
# Path & Key Utilities
# ─────────────────────────────────────────────────────────────────────────────

def default_lmdb_dir() -> Path:
    """Return default directory for the shared LMDB database."""
    return get_paths().cache_dir / DEFAULT_LMDB_NAME


def configured_lmdb_dir(log_callback: Callable[[str], None] | None = None) -> Path:
    """Retrieve LMDB path specified in settings.json or fallback to default."""
    paths = get_paths()
    settings = read_json(paths.settings_file, {}, log_callback)
    if isinstance(settings, dict):
        value = str(settings.get("lmdb_cache_dir", "")).strip()
        if value:
            return resolve_lmdb_dir(value)
    return default_lmdb_dir()


def resolve_lmdb_dir(value: str | Path | None) -> Path:
    """Normalize and resolve LMDB cache directory path."""
    text = str(value or "").strip()
    return Path(text).expanduser().resolve() if text else default_lmdb_dir()


def clean_rel_path(path: str | Path) -> str:
    """Normalize path separators to forward slashes for cross-platform LMDB keys."""
    return str(path).replace("\\", "/")


def path_key(rel_path: str | Path) -> bytes:
    """Key encoding for relative path indexing in LMDB (used by GUI and pipeline)."""
    return f"path:{clean_rel_path(rel_path)}".encode("utf-8")


def _path_key(rel_path: str) -> bytes:
    return path_key(rel_path)


def _train_key(index: int) -> bytes:
    """Key encoding for numeric index in LMDB (used by DINOv2 adapter training)."""
    return f"img:{index}".encode("ascii")


def _disk_free_bytes(path: Path) -> int:
    """Return free disk space in bytes for target drive."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def _lmdb_unavailable(message: str, mode: str, log_callback: Callable[[str], None] | None = None) -> None:
    if mode == "on":
        raise RuntimeError(message)
    emit_log(log_callback, message + " Continuing without LMDB.")


# ─────────────────────────────────────────────────────────────────────────────
# LMDB Hashing & Sizing
# ─────────────────────────────────────────────────────────────────────────────

def image_lmdb_hash(image_paths: list[Path], source_dir: Path) -> str:
    """Compute an MD5 fingerprint of the dataset based on relative paths, sizes, and mtimes."""
    records = []
    for path in sorted((Path(item) for item in image_paths), key=lambda p: clean_rel_path(p)):
        try:
            rel_path = clean_rel_path(path.resolve().relative_to(source_dir.resolve()))
        except Exception:
            rel_path = clean_rel_path(path.name)
        try:
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        records.append((rel_path, size, mtime))
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _extract_lmdb_hash(image_paths: list[str], data_dir: str) -> str:
    records = []
    for path in image_paths:
        try:
            stat = os.stat(path)
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        records.append((clean_rel_path(os.path.relpath(path, start=data_dir)), size, mtime))
    return hashlib.md5(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _train_lmdb_hash(samples: list[dict[str, str]]) -> str:
    records = []
    for sample in samples:
        path = sample["image_path"]
        try:
            stat = os.stat(path)
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        records.append((path, size, mtime, sample["description"]))
    return hashlib.md5(json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _lmdb_map_size(image_paths: list[Path], lmdb_dir: Path) -> int:
    total = 0
    for path in image_paths:
        try:
            total += path.stat().st_size
        except OSError:
            pass
    data_file = lmdb_dir / "data.mdb"
    existing_size = data_file.stat().st_size if data_file.exists() else 0
    return max(
        256 << 20,
        existing_size + int(total * 1.15) + (512 << 20),
        int(total * 1.25) + (512 << 20),
    )


def _cli_lmdb_map_size(image_paths: list[str]) -> int:
    total = 0
    for path in image_paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return max(128 << 20, int(total * 2.2) + (64 << 20))


# ─────────────────────────────────────────────────────────────────────────────
# LMDB Reader & Builder
# ─────────────────────────────────────────────────────────────────────────────

class ImageLmdbReader:
    """Thread-safe, high-speed read-only LMDB reader for cached images."""

    def __init__(self, lmdb_dir: str | Path, source_dir: str | Path):
        self.lmdb_dir = Path(lmdb_dir)
        self.source_dir = Path(source_dir).expanduser().resolve()
        self._env = None

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open(self):
        if lmdb is None:
            return None
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_dir),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
        return self._env

    def rel_path(self, path: str | Path) -> str:
        item = Path(path).expanduser().resolve()
        try:
            return clean_rel_path(item.relative_to(self.source_dir))
        except Exception:
            return clean_rel_path(item.name)

    def read_bytes(self, path: str | Path) -> bytes | None:
        env = self._open()
        if env is None:
            return None
        with env.begin(write=False) as txn:
            value = txn.get(path_key(self.rel_path(path)))
        return bytes(value) if value is not None else None

    def open_image(self, path: str | Path) -> Image.Image:
        """Fetch raw bytes from LMDB and decode as RGB PIL Image; fallback to disk on miss."""
        payload = self.read_bytes(path)
        if payload is None:
            with Image.open(path) as raw:
                return raw.convert("RGB")
        with Image.open(io.BytesIO(payload)) as raw:
            return raw.convert("RGB")


def prepare_image_lmdb(
    image_paths: list[Path],
    source_dir: str | Path,
    lmdb_dir: str | Path | None,
    mode: str = "auto",
    rebuild: bool = False,
    log_callback: Callable[[str], None] | None = None,
) -> ImageLmdbReader | None:
    """Build or verify an LMDB cache of uncompressed image binaries for ultra-fast I/O."""
    mode = str(mode or "auto").strip().lower()
    if mode == "off":
        return None
    if lmdb is None:
        message = "LMDB is not installed. Install lmdb or switch LMDB cache to Off."
        if mode == "on":
            raise RuntimeError(message)
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None

    source = Path(source_dir).expanduser().resolve()
    cache_dir = resolve_lmdb_dir(lmdb_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sorted_paths = sorted([Path(path).expanduser().resolve() for path in image_paths], key=lambda p: clean_rel_path(p))
    expected_hash = image_lmdb_hash(sorted_paths, source)
    meta_path = cache_dir / "extract_meta.json"

    if not rebuild and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("hash") == expected_hash and int(meta.get("count", 0)) == len(sorted_paths):
                emit_log(log_callback, f"LMDB image cache ready: {cache_dir}")
                return ImageLmdbReader(cache_dir, source)
        except Exception as exc:
            emit_log(log_callback, f"Warning: LMDB metadata is invalid, rebuilding: {exc}")

    map_size = _lmdb_map_size(sorted_paths, cache_dir)
    free_bytes = _disk_free_bytes(cache_dir)
    if free_bytes and map_size > int(free_bytes * 0.9):
        message = (
            f"LMDB cache directory does not have enough free space: {cache_dir}. "
            f"Need up to {format_size(map_size)}, free {format_size(free_bytes)}. "
            "Choose a larger Cache dir or switch LMDB to Off."
        )
        if mode == "on":
            raise RuntimeError(message)
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None

    emit_log(log_callback, f"Building LMDB image cache: {cache_dir} ({len(sorted_paths)} images, map={format_size(map_size)})")
    try:
        env = lmdb.open(
            str(cache_dir),
            map_size=map_size,
            subdir=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
    except Exception as exc:
        message = f"LMDB could not be opened at {cache_dir}: {exc}"
        if mode == "on":
            raise RuntimeError(message) from exc
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None
    txn = env.begin(write=True)
    written = skipped = 0
    try:
        for index, path in enumerate(sorted_paths):
            try:
                rel_path = clean_rel_path(path.relative_to(source))
            except Exception:
                rel_path = clean_rel_path(path.name)
            try:
                txn.put(path_key(rel_path), path.read_bytes())
                written += 1
            except OSError:
                skipped += 1
            if index > 0 and index % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
    except Exception:
        txn.abort()
        env.close()
        message = f"LMDB build failed at {cache_dir}. Choose a larger Cache dir or switch LMDB to Off."
        if mode == "on":
            raise RuntimeError(message)
        emit_log(log_callback, message + " Continuing without LMDB.")
        return None
    env.sync()
    env.close()

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "hash": expected_hash,
                "count": len(sorted_paths),
                "written": written,
                "skipped": skipped,
                "source_data_dir": str(source),
                "created_by": "main_app",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    emit_log(log_callback, f"LMDB image cache built: written={written}, skipped={skipped}")
    return ImageLmdbReader(cache_dir, source)


def prepare_extract_lmdb(image_paths: list[str], data_dir: str, lmdb_dir: str, mode: str, rebuild: bool = False):
    """CLI version of LMDB preparation for extract_tags.py."""
    if mode == "off":
        return None
    if lmdb is None:
        message = "LMDB unavailable: install the `lmdb` package."
        if mode == "on":
            raise RuntimeError(message)
        print(f"[LMDB] {message} Continuing without LMDB.")
        return None

    lmdb_path = os.path.abspath(lmdb_dir)
    os.makedirs(lmdb_path, exist_ok=True)
    sorted_paths = sorted([os.path.abspath(p) for p in image_paths], key=lambda p: clean_rel_path(p))
    expected_hash = _extract_lmdb_hash(sorted_paths, data_dir)
    meta_path = os.path.join(lmdb_path, "extract_meta.json")

    if not rebuild and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("hash") == expected_hash and int(meta.get("count", 0)) == len(sorted_paths):
                print(f"[LMDB] Cache ready: {lmdb_path} ({len(sorted_paths)} images)")
                return lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=256)
        except Exception as e:
            print(f"[LMDB] Cache metadata read error ({e}), rebuilding...")

    map_size = _cli_lmdb_map_size(sorted_paths)
    print(f"[LMDB] Building cache: {lmdb_path} ({len(sorted_paths)} images, map_size={format_size(map_size)})...")
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False, max_readers=256)
    txn = env.begin(write=True)
    written = skipped = 0
    for idx, img_path in enumerate(sorted_paths):
        rel = clean_rel_path(os.path.relpath(img_path, start=data_dir))
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            txn.put(_path_key(rel), img_bytes)
            written += 1
        except OSError:
            skipped += 1
        if idx > 0 and idx % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.sync()
    env.close()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hash": expected_hash,
                "count": len(sorted_paths),
                "written": written,
                "skipped": skipped,
                "data_dir": data_dir,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[LMDB] Cache build complete: written={written}, skipped={skipped}")
    return lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=256)


def read_lmdb_image(env, rel_path: str) -> bytes | None:
    """Read binary image bytes by relative path from an open LMDB environment."""
    if env is None:
        return None
    with env.begin(write=False) as txn:
        val = txn.get(_path_key(rel_path))
    return bytes(val) if val is not None else None


def write_train_lmdb(samples: list[dict[str, str]], data_dir: str, lmdb_dir: str, mode: str) -> None:
    """Write img:N index keys into LMDB to accelerate train_dinov.py training epochs."""
    if mode == "off" or lmdb is None or not samples:
        return

    lmdb_path = os.path.abspath(lmdb_dir)
    os.makedirs(lmdb_path, exist_ok=True)
    expected_hash = _train_lmdb_hash(samples)
    meta_path = os.path.join(lmdb_path, "train_meta.json")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("hash") == expected_hash and int(meta.get("count", 0)) == len(samples):
                return
        except Exception:
            pass

    image_paths = [sample["image_path"] for sample in samples]
    map_size = _cli_lmdb_map_size(image_paths)
    env = lmdb.open(lmdb_path, map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False, max_readers=256)
    txn = env.begin(write=True)
    written = skipped = 0
    try:
        for idx, sample in enumerate(samples):
            img_path = sample["image_path"]
            rel = clean_rel_path(os.path.relpath(img_path, start=data_dir))
            img_bytes = txn.get(_path_key(rel))
            if img_bytes is None:
                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    txn.put(_path_key(rel), img_bytes)
                except OSError:
                    skipped += 1
                    continue
            txn.put(_train_key(idx), img_bytes)
            written += 1
            if idx > 0 and idx % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.sync()
    finally:
        env.close()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hash": expected_hash,
                "count": len(samples),
                "written": written,
                "skipped": skipped,
                "data_dir": data_dir,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset & Directory Scanning
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetRecord:
    path: Path
    image: Image.Image | None
    error: str | None = None


class ImagePathDataset(Dataset):
    """PyTorch Dataset that loads and generates 512x512 thumbnails directly from LMDB or disk."""

    def __init__(self, paths: list[Path], lmdb_dir: str | Path | None = None, lmdb_source: str | Path | None = None):
        self.paths = [Path(path) for path in paths]
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
        return len(self.paths)

    def __getitem__(self, idx: int) -> DatasetRecord:
        path = self.paths[idx]
        try:
            reader = self._reader()
            if reader is None:
                with Image.open(path) as raw:
                    image = raw.convert("RGB")
            else:
                image = reader.open_image(path)
            image.thumbnail((512, 512), Image.Resampling.BILINEAR)
            return DatasetRecord(path=path, image=image)
        except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as exc:
            return DatasetRecord(path=path, image=None, error=str(exc))


def pil_collate(batch: list[DatasetRecord]) -> list[DatasetRecord]:
    """Identity collation function returning batch of PIL DatasetRecords."""
    return batch


def parse_blocked_subfolders(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Parse comma, semicolon, or newline delimited blocked subfolder filters."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = re.split(r"[\n;,]+", str(value))
    blocked: list[str] = []
    for item in raw_items:
        cleaned = item.strip().strip('"').strip("'")
        if cleaned:
            blocked.append(cleaned)
    return blocked


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _blocked_matchers(source_path: Path, blocked_subfolders: str | list[str] | tuple[str, ...] | None) -> tuple[list[Path], list[str]]:
    absolute_or_relative_paths: list[Path] = []
    names: list[str] = []
    for item in parse_blocked_subfolders(blocked_subfolders):
        normalized = item.replace("\\", "/").strip("/")
        path_item = Path(item).expanduser()
        if path_item.is_absolute():
            absolute_or_relative_paths.append(path_item.resolve())
        elif "/" in normalized:
            absolute_or_relative_paths.append((source_path / normalized).resolve())
        else:
            names.append(normalized.casefold())
    return absolute_or_relative_paths, names


def _is_blocked_path(path: Path, blocked_paths: list[Path], blocked_names: list[str]) -> bool:
    resolved = path.resolve()
    if any(_is_relative_to(resolved, blocked_path) for blocked_path in blocked_paths):
        return True
    if blocked_names and any(part.casefold() in blocked_names for part in resolved.parts):
        return True
    return False


def scan_images(
    source: str | Path,
    recursive: bool,
    blocked_subfolders: str | list[str] | tuple[str, ...] | None = None,
    cache_status_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> list[Path]:
    """Scan source folder for supported images with mtime cache validation and blocked path filtering."""
    source_path = Path(source).expanduser().resolve()
    paths_cfg = get_paths()
    cache_file = paths_cfg.scan_cache_file
    try:
        mtime = str(source_path.stat().st_mtime)
    except OSError:
        mtime = "0"
    blocked_paths, blocked_names = _blocked_matchers(source_path, blocked_subfolders)
    blocked_key = "|".join([str(path) for path in blocked_paths] + blocked_names)
    cache_key = f"{source_path}|{recursive}|{mtime}|blocked={blocked_key}"

    cache = read_json(cache_file, {}, log_callback)
    if cache.get("key") == cache_key:
        cached_paths = [Path(item) for item in cache.get("paths", []) if Path(item).is_file()]
        if cache_status_callback is not None:
            cache_status_callback(f"cached ({len(cached_paths)})")
        return cached_paths

    if cache_status_callback is not None:
        cache_status_callback("scanning...")
    if blocked_paths or blocked_names:
        emit_log(log_callback, f"Blocked subfolders: {len(blocked_paths) + len(blocked_names)}")

    if recursive:
        found_paths: list[Path] = []
        for root, dirs, files in os.walk(source_path):
            root_path = Path(root)
            dirs[:] = [
                dirname
                for dirname in dirs
                if not _is_blocked_path(root_path / dirname, blocked_paths, blocked_names)
            ]
            for filename in files:
                path = root_path / filename
                if path.suffix.lower() in VALID_EXTS and not _is_blocked_path(path, blocked_paths, blocked_names):
                    found_paths.append(path)
    else:
        found_paths = [
            path
            for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTS and not _is_blocked_path(path, blocked_paths, blocked_names)
        ]

    payload = {"key": cache_key, "paths": [str(path) for path in found_paths]}
    write_json(paths_cfg.cache_dir / "scan_cache.json", payload, log_callback)
    if cache_status_callback is not None:
        cache_status_callback(f"scanned ({len(found_paths)})")
    return found_paths


def clear_cache(log_callback: Callable[[str], None] | None = None, lmdb_dir: str | Path | None = None) -> tuple[list[str], str | None]:
    """Delete scan_cache.json, feat_cache.pt, and LMDB database directory."""
    paths = get_paths()
    removed: list[str] = []
    error: str | None = None
    for label, path in (
        ("scan", paths.resolve_cache_file("scan_cache.json")),
        ("features", paths.resolve_cache_file("feat_cache.pt")),
    ):
        try:
            if path.exists():
                size = path.stat().st_size if label == "features" else 0
                path.unlink()
                removed.append(f"{label} ({format_size(size)})" if label == "features" else label)
        except Exception as exc:
            error = str(exc)
            emit_log(log_callback, f"Warning: failed to clear cache file {path.name}: {exc}")
    lmdb_paths = [paths.cache_dir / "dinov2_train_lmdb"]
    if lmdb_dir:
        custom = Path(lmdb_dir).expanduser()
        if custom not in lmdb_paths:
            lmdb_paths.append(custom)
    for path in lmdb_paths:
        try:
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                removed.append("lmdb")
        except Exception as exc:
            error = str(exc)
            emit_log(log_callback, f"Warning: failed to clear LMDB cache {path}: {exc}")
    return removed, error
