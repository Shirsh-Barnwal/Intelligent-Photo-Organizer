from __future__ import annotations

# Backward compatibility alias - all functionality moved to app.caching
from .caching import (
    DEFAULT_LMDB_NAME,
    ImageLmdbReader,
    clean_rel_path,
    configured_lmdb_dir,
    default_lmdb_dir,
    image_lmdb_hash,
    path_key,
    prepare_image_lmdb,
    resolve_lmdb_dir,
)

__all__ = [
    "DEFAULT_LMDB_NAME",
    "default_lmdb_dir",
    "configured_lmdb_dir",
    "resolve_lmdb_dir",
    "clean_rel_path",
    "path_key",
    "image_lmdb_hash",
    "ImageLmdbReader",
    "prepare_image_lmdb",
]
