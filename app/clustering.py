from __future__ import annotations

import logging
import re
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .utils import emit_log, emit_progress, safe_filename

logger = logging.getLogger(__name__)


def natural_sort_key(path: Path) -> tuple[str, list[Any]]:
    """Sort filenames naturally respecting embedded numbers (e.g., page1, page2, page10)."""
    parts = re.split(r"(\d+)", path.name.lower())
    normalized: list[Any] = []
    for part in parts:
        normalized.append(int(part) if part.isdigit() else part)
    return (str(path.parent), normalized)


def deduplicate_images(
    paths: list[Path],
    feat_list: list[torch.Tensor],
    subs: list[str],
    threshold: float,
) -> tuple[list[Path], list[torch.Tensor], list[str]]:
    """Identify and filter out near-duplicate images based on cosine similarity threshold."""
    if not feat_list:
        return paths, feat_list, subs

    feats = F.normalize(torch.cat(feat_list, dim=0), p=2, dim=1).float()
    removed: set[int] = set()
    chunk_size = 512
    for index in range(len(paths)):
        if index in removed:
            continue
        remaining = feats[index + 1 :]
        if remaining.shape[0] == 0:
            break
        for chunk_start in range(0, remaining.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, remaining.shape[0])
            chunk = remaining[chunk_start:chunk_end]
            sims = torch.mm(feats[index : index + 1], chunk.t())[0]
            dup_indices = (sims >= threshold).nonzero(as_tuple=True)[0].tolist()
            for dup_index in dup_indices:
                removed.add(index + 1 + chunk_start + dup_index)

    keep = [index for index in range(len(paths)) if index not in removed]
    return [paths[index] for index in keep], [feat_list[index] for index in keep], [subs[index] for index in keep]


def cluster_images(
    paths: list[Path],
    feat_list: list[torch.Tensor],
    subs: list[str],
    sensitivity: float,
    max_clusters: int = 0,
    log_callback: Callable[[str], None] | None = None,
) -> dict[int, list[tuple[Path, str]]]:
    """Perform visual clustering using UMAP dimensionality reduction + Agglomerative Hierarchical Clustering."""
    if not feat_list:
        return {}
    if len(feat_list) == 1:
        emit_log(log_callback, "  Found 1 group (only 1 image)")
        return {0: [(paths[0], subs[0])]}

    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import euclidean_distances

    full_feats_tensor = F.normalize(torch.cat(feat_list, dim=0), p=2, dim=1).float().cpu()
    feats = full_feats_tensor.detach().numpy()
    n_samples = feats.shape[0]
    feat_dim = feats.shape[1]
    emit_log(log_callback, f"  Features: {n_samples} samples x {feat_dim} dims")

    umap_dim = min(20, feat_dim - 1, max(2, n_samples - 2))
    if feat_dim > 32 and n_samples > umap_dim + 2:
        try:
            import umap

            n_neighbors = max(2, min(15, n_samples - 1))
            emit_log(log_callback, f"  UMAP: {feat_dim}d -> {umap_dim}d (neighbors={n_neighbors})")
            reducer = umap.UMAP(
                n_components=umap_dim,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            feats = reducer.fit_transform(feats)
            emit_log(log_callback, f"  UMAP done -> {feats.shape[1]}d")
        except Exception as exc:
            emit_log(log_callback, f"  Warning: UMAP failed ({exc}), using raw features")
            emit_log(log_callback, traceback.format_exc())

    dists = euclidean_distances(feats)
    tri_idx = np.triu_indices_from(dists, k=1)
    flat_dists = dists[tri_idx]
    threshold = float(np.percentile(flat_dists, sensitivity * 100))
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="euclidean",
        linkage="ward",
    ).fit_predict(feats)

    label_to_indices: dict[int, list[int]] = {}
    groups: dict[int, list[tuple[Path, str]]] = {}
    for index, (path, sub, label) in enumerate(zip(paths, subs, labels)):
        label_id = int(label)
        label_to_indices.setdefault(label_id, []).append(index)
        groups.setdefault(label_id, []).append((path, sub))

    emit_log(log_callback, f"  Found {len(groups)} clusters")
    if max_clusters > 0 and len(groups) > max_clusters:
        groups = merge_visual_clusters_by_similarity(
            groups=groups,
            label_to_indices=label_to_indices,
            feats=full_feats_tensor,
            max_clusters=max_clusters,
            log_callback=log_callback,
        )
    return groups


def merge_visual_clusters_by_similarity(
    *,
    groups: dict[int, list[tuple[Path, str]]],
    label_to_indices: dict[int, list[int]],
    feats: torch.Tensor,
    max_clusters: int,
    log_callback: Callable[[str], None] | None = None,
) -> dict[int, list[tuple[Path, str]]]:
    """Merge smaller overflow clusters into the nearest anchor clusters by centroid cosine similarity."""
    if max_clusters <= 0 or len(groups) <= max_clusters:
        return groups

    max_clusters = max(1, int(max_clusters))
    sorted_labels = sorted(groups, key=lambda label: len(groups[label]), reverse=True)
    anchor_labels = sorted_labels[:max_clusters]
    overflow_labels = sorted_labels[max_clusters:]

    def centroid_for(label: int) -> torch.Tensor:
        indices = label_to_indices.get(label, [])
        if not indices:
            return torch.zeros(feats.shape[1], dtype=feats.dtype)
        centroid = feats[indices].mean(dim=0)
        return F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

    anchor_centroids = {label: centroid_for(label) for label in anchor_labels}
    merged = {label: list(groups[label]) for label in anchor_labels}
    merged_indices = {label: list(label_to_indices.get(label, [])) for label in anchor_labels}
    reassigned_images = 0

    for label in overflow_labels:
        source_centroid = centroid_for(label)
        best_label = max(
            anchor_labels,
            key=lambda anchor: F.cosine_similarity(source_centroid, anchor_centroids[anchor], dim=0).item(),
        )
        merged[best_label].extend(groups[label])
        merged_indices[best_label].extend(label_to_indices.get(label, []))
        reassigned_images += len(groups[label])

        updated_indices = merged_indices[best_label]
        if updated_indices:
            anchor_centroids[best_label] = F.normalize(feats[updated_indices].mean(dim=0).unsqueeze(0), p=2, dim=1).squeeze(0)

    emit_log(
        log_callback,
        f"  Folder limit for visual grouping: {len(groups)} -> {len(merged)} clusters. "
        f"Reassigned {reassigned_images} images to nearest clusters.",
    )
    return merged


def apply_folder_limit(
    accepted: list[tuple[Path, str]],
    max_folders: int,
    feat_cache_file: Path,
    use_dino_adapter: bool,
    sanitize_subfolder_fn: Callable[[str], str],
    log_callback: Callable[[str], None] | None = None,
) -> list[tuple[Path, str]]:
    """Enforce a strict maximum limit on total created folders by remapping overflow images."""
    if max_folders <= 0:
        return accepted

    counts: dict[str, int] = {}
    for _path, sub in accepted:
        sanitized = sanitize_subfolder_fn(sub)
        if sanitized:
            counts[sanitized] = counts.get(sanitized, 0) + 1
    unique_folders = len(counts)
    if unique_folders <= max_folders:
        return accepted

    sorted_folders = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    keep_set = {name for name, _cnt in sorted_folders[:max_folders]}
    fallback_folder = sorted_folders[0][0]

    try:
        feat_cache = (
            torch.load(str(feat_cache_file), map_location="cpu", weights_only=True)
            if feat_cache_file.exists()
            else {}
        )
    except Exception:
        feat_cache = {}

    def _load_feat(path: Path) -> torch.Tensor | None:
        try:
            mtime = str(path.stat().st_mtime)
        except OSError:
            mtime = "0"

        dino_key = f"dinov2ms_{path}_{mtime}_{use_dino_adapter}"
        dino_val = feat_cache.get(dino_key)
        if isinstance(dino_val, torch.Tensor):
            vec = dino_val.view(-1).cpu().float()
            if vec.numel() > 0:
                return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)

        siglip_key = f"siglip_{path}_{mtime}"
        siglip_val = feat_cache.get(siglip_key)
        if isinstance(siglip_val, torch.Tensor):
            vec = siglip_val.view(-1).cpu().float()
            if vec.numel() > 0:
                return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
        return None

    folder_vectors: dict[str, list[torch.Tensor]] = {folder: [] for folder in keep_set}
    for path, sub in accepted:
        sanitized = sanitize_subfolder_fn(sub)
        if sanitized in keep_set:
            vec = _load_feat(path)
            if vec is not None:
                folder_vectors[sanitized].append(vec)

    folder_centroids: dict[str, torch.Tensor] = {}
    for folder, vectors in folder_vectors.items():
        if not vectors:
            continue
        centroid = torch.stack(vectors, dim=0).mean(dim=0)
        folder_centroids[folder] = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

    remapped: list[tuple[Path, str]] = []
    merged_count = 0
    similarity_routed = 0
    for path, sub in accepted:
        sanitized = sanitize_subfolder_fn(sub)
        if not sanitized:
            remapped.append((path, sub))
            continue
        if sanitized in keep_set:
            remapped.append((path, sanitized))
        else:
            vec = _load_feat(path)
            best_folder = None
            best_score = -1.0
            if vec is not None and folder_centroids:
                for folder, centroid in folder_centroids.items():
                    score = F.cosine_similarity(vec, centroid, dim=0).item()
                    if score > best_score:
                        best_score = score
                        best_folder = folder
            if best_folder:
                remapped.append((path, best_folder))
                similarity_routed += 1
            else:
                remapped.append((path, fallback_folder))
            merged_count += 1

    emit_log(
        log_callback,
        f"Folder limit applied: {unique_folders} -> {max_folders}. "
        f"Reassigned by similarity: {similarity_routed}. "
        f"Fallback merged: {merged_count - similarity_routed} -> '{fallback_folder}'.",
    )
    return remapped


def group_sequences(
    accepted: list[tuple[Path, str]],
    dino_paths: list[Path],
    dino_feats: list[torch.Tensor],
    dino_subs: list[str],
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> None:
    """Group sequential images (manga chapters, photo bursts) using filename stems, timestamps, and DINO features."""
    emit_log(log_callback, "Grouping sequential images (Manga mode with heuristics)...")

    def get_clean_base(name: str) -> str:
        base = Path(name).stem
        return re.sub(r"[\s_\-]*\d+$", "", base).lower()

    def get_prefix(name: str, length: int = 6) -> str:
        return Path(name).stem[:length].lower()

    if dino_feats:
        cpu_feats = [value.cpu().float() for value in dino_feats]
        sorted_idx = list(range(len(dino_paths)))
        sorted_idx.sort(key=lambda item: natural_sort_key(dino_paths[item]))

        dir_seq_index: dict[str, list[int]] = {}
        sequences: list[dict[str, Any]] = []

        with torch.inference_mode():
            for sorted_pos, item_index in enumerate(sorted_idx):
                if sorted_pos % 50 == 0:
                    emit_progress(progress_callback, 80 + (sorted_pos / max(1, len(sorted_idx))) * 5)

                path = dino_paths[item_index]
                dirname = str(path.parent)
                filename = path.name
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0

                base = get_clean_base(filename)
                prefix = get_prefix(filename)
                feat = cpu_feats[item_index]
                best_seq = None
                best_score = -1.0

                for seq_idx in dir_seq_index.get(dirname, [])[-20:]:
                    seq = sequences[seq_idx]
                    score = 0.0
                    if abs(mtime - seq["first_mtime"]) <= 40:
                        score += 0.6
                    if base and base == seq["base"]:
                        score += 0.4
                    if prefix and prefix == seq["prefix"]:
                        score += 0.3
                    sim = F.cosine_similarity(feat, seq["last_feat"], dim=-1).item()
                    if sim > 0.85:
                        score += 0.5
                    elif sim > 0.70:
                        score += 0.2

                    if score >= 0.7 and score > best_score:
                        best_score = score
                        best_seq = seq

                if best_seq is not None:
                    best_seq["indices"].append(item_index)
                    best_seq["last_feat"] = feat
                    best_seq["last_mtime"] = mtime
                else:
                    new_seq = {
                        "indices": [item_index],
                        "base": base,
                        "prefix": prefix,
                        "first_mtime": mtime,
                        "last_mtime": mtime,
                        "last_feat": feat,
                    }
                    seq_idx = len(sequences)
                    sequences.append(new_seq)
                    dir_seq_index.setdefault(dirname, []).append(seq_idx)

        unified_count = 0
        for seq in sequences:
            if len(seq["indices"]) > 1:
                subs_in_seq = [dino_subs[idx] for idx in seq["indices"] if dino_subs[idx]]
                if subs_in_seq:
                    best_sub = max(set(subs_in_seq), key=subs_in_seq.count)
                    for idx in seq["indices"]:
                        if dino_subs[idx] != best_sub:
                            dino_subs[idx] = best_sub
                            unified_count += 1

        num_seqs = sum(1 for item in sequences if len(item["indices"]) > 1)
        emit_log(log_callback, f"  Unified folders for {num_seqs} sequences ({unified_count} images)")
