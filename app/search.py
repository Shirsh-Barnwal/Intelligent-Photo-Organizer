from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .utils import emit_log, get_paths, safe_filename

logger = logging.getLogger(__name__)


def run_ai_search(
    config: dict[str, Any],
    progress_callback: Callable[[float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    manager: Any | None = None,
) -> dict[str, Any]:
    """Execute local zero-shot semantic search over images using cached SigLIP2 embeddings."""
    if manager is None:
        from .models import ModelManager

        manager = ModelManager()

    manager.set_callbacks(progress_callback, log_callback)
    manager.update_runtime(config)

    search_query = str(config.get("query", "")).strip()
    if not search_query:
        return {"ok": False, "message": "Search query cannot be empty."}

    target_dir = Path(config.get("target_dir", ".")).expanduser().resolve()
    top_k = int(config.get("top_k", 20))
    copy_results = bool(config.get("copy_results", True))
    selected_paths_raw = config.get("selected_paths")
    selected_paths = (
        {str(Path(path).expanduser().resolve()) for path in selected_paths_raw}
        if isinstance(selected_paths_raw, list)
        else None
    )

    if not manager.ensure_siglip_loaded():
        return {"ok": False, "message": "SigLIP2 failed to load."}

    feat_cache_file = get_paths().feat_cache_file
    if not feat_cache_file.exists():
        return {"ok": False, "message": f"No feature cache found at {feat_cache_file}. Run a scan or sort first."}

    try:
        feat_cache = torch.load(str(feat_cache_file), map_location="cpu", weights_only=True)
        siglip_cache = {key: value for key, value in feat_cache.items() if str(key).startswith("siglip_")}
        if not siglip_cache:
            return {"ok": False, "message": "No SigLIP embeddings found in cache."}

        with torch.inference_mode():
            query_feat = manager.siglip_embed_texts([search_query])[0].cpu()

        paths: list[Path] = []
        feats: list[torch.Tensor] = []
        for key, value in siglip_cache.items():
            without_prefix = str(key)[len("siglip_") :]
            last_us = without_prefix.rfind("_")
            path_text = without_prefix[:last_us] if last_us > 0 else without_prefix
            paths.append(Path(path_text))
            feats.append(value.view(1, -1))

        all_feats = F.normalize(torch.cat(feats, dim=0), p=2, dim=1)
        sims = torch.mm(all_feats, query_feat.unsqueeze(1)).squeeze(1)
        top_k = min(top_k, len(paths))
        scores, indices = torch.topk(sims, top_k)

        result_items: list[dict[str, Any]] = []
        for index, item in enumerate(indices):
            source_path = paths[item.item()]
            score = float(scores[index].item())
            result_items.append(
                {
                    "rank": index + 1,
                    "path": str(source_path),
                    "score": score,
                    "exists": source_path.exists(),
                }
            )

        copied = 0
        result_dir: Path | None = None
        if copy_results:
            result_dir = target_dir / "AI_Search_Results" / safe_filename(search_query)[:50]
            result_dir.mkdir(parents=True, exist_ok=True)
            for item in result_items:
                source_path = Path(str(item["path"]))
                if not source_path.exists():
                    continue
                if selected_paths is not None and str(source_path.resolve()) not in selected_paths:
                    continue
                destination = result_dir / f"{int(item['rank']):03d}_{float(item['score']):.2f}{source_path.suffix}"
                shutil.copy2(source_path, destination)
                copied += 1

        return {
            "ok": True,
            "message": f"Copied {copied} files to {result_dir}" if copy_results else f"Found {len(result_items)} matches",
            "result_dir": str(result_dir) if result_dir is not None else "",
            "copied": copied,
            "results": result_items,
        }
    except Exception as exc:
        emit_log(log_callback, f"AI search failed: {exc}")
        emit_log(log_callback, traceback.format_exc())
        return {"ok": False, "message": str(exc)}
