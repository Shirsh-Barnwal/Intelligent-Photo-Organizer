from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .caching import clean_rel_path, configured_lmdb_dir, prepare_extract_lmdb, read_lmdb_image, write_train_lmdb
from .utils import (
    VALID_EXTS,
    emit_log,
    format_size,
    get_paths,
    read_json,
    safe_filename,
)

logger = logging.getLogger(__name__)

# Predefined tag groups for semantic state vector extraction
WD_NUDE_TAGS = {"nude", "unskirt", "nipples", "pussy", "penis", "breasts", "cleavage", "underwear", "bikini", "swimsuit", "naked"}
WD_P_STAND_TAGS = {"standing", "on_one_leg"}
WD_P_SIT_TAGS = {"sitting", "kneeling", "squatting", "seiza", "crossed_legs"}
WD_P_LIE_TAGS = {"lying", "on_stomach", "on_back", "on_side"}
WD_C_SHIRT_TAGS = {"shirt", "t-shirt", "collared_shirt", "blouse", "tank_top", "sweater", "hoodie"}
WD_C_JACKET_TAGS = {"jacket", "coat", "cardigan", "suit_jacket", "cloak"}
WD_C_DRESS_TAGS = {"dress", "sundress", "wedding_dress"}
WD_C_SKIRT_TAGS = {"skirt", "pleated_skirt", "miniskirt", "pencil_skirt"}
WD_C_PANTS_TAGS = {"pants", "jeans", "shorts", "sweatpants", "trousers", "leggings"}
WD_SC_INDOOR = {"indoors", "bedroom", "classroom", "office", "kitchen", "bathroom", "living_room"}
WD_SC_OUTDOOR = {"outdoors", "sky", "cloud", "street", "road", "bridge"}
WD_SC_NATURE = {"nature", "forest", "mountain", "ocean", "lake", "river", "field", "tree", "flower"}
WD_SC_URBAN = {"city", "building", "skyscraper", "cityscape", "town", "alley"}
WD_H_BLACK = {"black_hair"}
WD_H_BLONDE = {"blonde_hair", "light_brown_hair"}
WD_H_BROWN = {"brown_hair"}
WD_H_RED = {"red_hair", "pink_hair"}
WD_H_BLUE = {"blue_hair", "aqua_hair"}
WD_H_WHITE = {"white_hair", "grey_hair", "silver_hair"}
WD_H_GREEN = {"green_hair"}
WD_H_PURPLE = {"purple_hair"}
WD_EX_HAPPY = {"smile", "grin", "laughing", "open_mouth", ":d"}
WD_EX_SAD = {"crying", "tears", "frown", "sad"}
WD_EX_ANGRY = {"angry", "furrowed_brow", "clenched_teeth"}
WD_EX_NEUTRAL = {"expressionless", "closed_mouth", "serious"}


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing for ONNX Taggers
# ─────────────────────────────────────────────────────────────────────────────

def prepare_image(image: Image.Image, engine: str = "wd", target_size: int = 448) -> np.ndarray:
    """Preprocess PIL image for ONNX inference matching WD or Camie tagger input specs."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
    if engine.lower().startswith("camie"):
        import torchvision.transforms as transforms

        aspect_ratio = w / h
        if aspect_ratio > 1:
            nw = 512
            nh = int(512 / aspect_ratio)
        else:
            nh = 512
            nw = int(512 * aspect_ratio)

        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        pad_color = (124, 116, 104)
        padded = Image.new("RGB", (512, 512), pad_color)
        padded.paste(image, ((512 - nw) // 2, (512 - nh) // 2))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transform(padded).unsqueeze(0).numpy()
    else:
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        padded = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        padded.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))

        arr = np.array(padded, dtype=np.float32) / 255.0
        arr = arr[:, :, ::-1].copy()  # RGB to BGR
        return np.expand_dims(arr, axis=0).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Semantic State Vector Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_semantic_vector(
    preds: dict[str, float],
    wd_tag_categories: dict[str, Any] | None,
    log_callback: Callable[[str], None] | None = None,
) -> torch.Tensor | None:
    """Construct a combined semantic state vector (nude/pose/scene/expression + character hash)."""
    try:
        def get_max_score(tag_set: set[str]) -> float:
            return max([score for tag, score in preds.items() if tag in tag_set] + [0.0])

        nude_score = get_max_score(WD_NUDE_TAGS)
        state_vec = torch.tensor(
            [
                nude_score,
                max(0.0, 1.0 - nude_score),
                get_max_score(WD_P_STAND_TAGS),
                get_max_score(WD_P_SIT_TAGS),
                get_max_score(WD_P_LIE_TAGS),
                get_max_score(WD_C_SHIRT_TAGS),
                get_max_score(WD_C_JACKET_TAGS),
                get_max_score(WD_C_DRESS_TAGS),
                get_max_score(WD_C_SKIRT_TAGS),
                get_max_score(WD_C_PANTS_TAGS),
                get_max_score(WD_SC_INDOOR),
                get_max_score(WD_SC_OUTDOOR),
                get_max_score(WD_SC_NATURE),
                get_max_score(WD_SC_URBAN),
                get_max_score(WD_H_BLACK),
                get_max_score(WD_H_BLONDE),
                get_max_score(WD_H_BROWN),
                get_max_score(WD_H_RED),
                get_max_score(WD_H_BLUE),
                get_max_score(WD_H_WHITE),
                get_max_score(WD_H_GREEN),
                get_max_score(WD_H_PURPLE),
                get_max_score(WD_EX_HAPPY),
                get_max_score(WD_EX_SAD),
                get_max_score(WD_EX_ANGRY),
                get_max_score(WD_EX_NEUTRAL),
            ],
            dtype=torch.float32,
            device="cpu",
        )

        char_preds = [
            (tag, score)
            for tag, score in preds.items()
            if wd_tag_categories and wd_tag_categories.get(tag, -1) in (4, "character")
        ]
        top_chars = sorted(char_preds, key=lambda item: item[1], reverse=True)[:8]
        char_dim = 38
        char_vec = torch.zeros(char_dim, device="cpu")
        for tag, score in top_chars:
            if score > 0.35:
                index = hash(tag) % char_dim
                char_vec[index] = max(char_vec[index].item(), score)
        return torch.cat([state_vec, char_vec], dim=-1).unsqueeze(0)
    except Exception as exc:
        emit_log(log_callback, f"Warning: failed to build semantic vector: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tag Normalization & Folder String Utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalized_tag(tag: str) -> str:
    """Normalize tags with snake_case and resolve common tag aliases."""
    cleaned = re.sub(r"[\s\-]+", "_", str(tag).strip().lower())
    aliases = {
        "1_girl": "1girl",
        "solo_female": "1girl",
        "solo_male": "1boy",
        "multiple_girl": "multiple_girls",
        "multiple_boy": "multiple_boys",
        "kitchenknife": "kitchen_knife",
        "pinkapron": "pink_apron",
        "sea": "ocean",
        "seaside": "beach",
        "indoors_scene": "indoors",
        "outdoors_scene": "outdoors",
    }
    return aliases.get(cleaned, cleaned)


def display_tag(tag: str) -> str:
    """Format tags for clean user-facing folder names (e.g., '1Girl', 'Blue Hair')."""
    token = str(tag).strip().lower()
    special = {
        "1girl": "1Girl",
        "1boy": "1Boy",
        "2girls": "2Girls",
        "2boys": "2Boys",
        "multiple_girls": "Multiple Girls",
        "multiple_boys": "Multiple Boys",
    }
    if token in special:
        return special[token]
    return token.replace("_", " ").title()


def folder_name_tokens(value: str) -> set[str]:
    """Tokenize folder name for similarity comparisons."""
    text = re.sub(r"[_\\/]+", " ", str(value).lower())
    return {token for token in re.findall(r"[a-zа-яё0-9]+", text) if token}


def folder_name_similarity(left: str, right: str) -> float:
    """Measure lexical overlap between folder names using Jaccard & containment."""
    left_tokens = folder_name_tokens(left)
    right_tokens = folder_name_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = intersection / max(union, 1)
    containment = max(
        intersection / max(len(left_tokens), 1),
        intersection / max(len(right_tokens), 1),
    )
    left_text = " ".join(sorted(left_tokens))
    right_text = " ".join(sorted(right_tokens))
    substring = 1.0 if left_text in right_text or right_text in left_text else 0.0
    return max(jaccard, containment * 0.9, substring)


def generic_folder_name_tokens() -> set[str]:
    """Tokens considered generic or non-distinctive across image clusters."""
    return {
        "1girl",
        "1boy",
        "2girls",
        "2boys",
        "girl",
        "girls",
        "boy",
        "boys",
        "female",
        "male",
        "character",
        "solo",
        "portrait",
        "close",
        "up",
        "smile",
        "smiling",
        "looking",
        "viewer",
        "standing",
        "sitting",
        "long",
        "short",
        "hair",
        "shirt",
        "white",
        "black",
        "background",
        "simple",
        "indoors",
        "outdoors",
        "person",
        "people",
    }


def canonical_folder_name(value: str) -> str:
    """Normalize cluster folder names to canonical form ('Female Character', etc.)."""
    tokens = folder_name_tokens(value)
    parts: list[str] = []
    if {"1girl", "girl", "girls", "woman", "female"} & tokens:
        parts.append("Female Character")
    elif {"1boy", "boy", "boys", "man", "male"} & tokens:
        parts.append("Male Character")
    cleaned = str(value)
    cleaned = re.sub(r"\b(?:1girl|girl|girls|woman|female|1boy|boy|boys|man|male)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        parts.append(cleaned)
    return safe_filename(" ".join(parts) if parts else value)


def hierarchical_folder_name(value: str) -> str:
    """Organize names into top-level hierarchical categories ('Character/...', 'Scene/...')."""
    tokens = folder_name_tokens(value)
    if {"female", "male", "character", "1girl", "1boy", "girl", "boy"} & tokens:
        return str(Path("Character") / safe_filename(value))
    if {"beach", "forest", "city", "street", "classroom", "kitchen", "bedroom", "outdoors", "indoors"} & tokens:
        return str(Path("Scene") / safe_filename(value))
    if {"sketch", "watercolor", "pixel", "anime", "photo", "render"} & tokens:
        return str(Path("Style") / safe_filename(value))
    return value


def display_character_name(tag: str) -> str:
    """Clean anime/character tag names (removing parentheses/series info)."""
    token = normalized_tag(tag)
    token = re.sub(r"_\([^)]*\)$", "", token)
    token = re.sub(r"\([^)]*\)$", "", token)
    token = token.replace("_", " ").strip()
    if not token:
        return ""
    return safe_filename(token.title())


def contains_year_token(value: str) -> bool:
    """Check if a string contains year tokens like 1999, 2024, or 'years'."""
    text = str(value).lower()
    return bool(
        re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|2100)(?!\d)", text)
        or re.search(r"\byears?\b", text)
        or re.search(r"\byear[_\-\s]*(?:19\d{2}|20\d{2}|2100)\b", text)
    )


def strip_year_tokens(value: str) -> str:
    """Strip chronological tags and year indicators from cluster/folder names."""
    cleaned = str(value)
    cleaned = re.sub(r"\byear[_\-\s]*(?:19\d{2}|20\d{2}|2100)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:19\d{2}|20\d{2}|2100)[_\-\s]*years?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<!\d)(?:19\d{2}|20\d{2}|2100)(?!\d)", " ", cleaned)
    cleaned = re.sub(r"\byears?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-()[]{}")
    return cleaned


def sanitize_subfolder_path(sub: str) -> str:
    """Clean and strip year tokens across path segments."""
    if not sub:
        return ""
    parts = [strip_year_tokens(part) for part in re.split(r"[\\/]+", sub) if part]
    parts = [safe_filename(part) for part in parts if part]
    return str(Path(*parts)) if parts else ""


# ─────────────────────────────────────────────────────────────────────────────
# Character-Aware Routing
# ─────────────────────────────────────────────────────────────────────────────

def source_character_hint(path: Path, source_dir: str | Path, recursive_scan: bool) -> str:
    """Extract character hint from source subfolder name when recursive scanning is active."""
    if not recursive_scan:
        return ""
    try:
        source = Path(source_dir).expanduser().resolve()
        rel_parent = path.resolve().parent.relative_to(source)
    except Exception:
        return ""
    parts = [part for part in rel_parent.parts if part and part not in {".", ".."}]
    if not parts:
        return ""
    ignored = {
        "images",
        "image",
        "img",
        "pics",
        "pictures",
        "photos",
        "new",
        "old",
        "misc",
        "other",
        "sorted",
        "unsorted",
        "unknown",
        "download",
        "downloads",
    }
    for part in parts:
        cleaned = safe_filename(str(part).replace("_", " ").replace("-", " "))
        if cleaned and cleaned.lower() not in ignored and not contains_year_token(cleaned):
            return cleaned.title()
    return ""


def character_folder_for_image(
    path: Path,
    preds: dict[str, float] | None,
    config: dict[str, Any],
    wd_tag_categories: dict[str, Any] | None,
) -> str:
    """Determine character folder assignment from WD predictions and confidence margins."""
    if not (config.get("character_aware_recursive") and config.get("recursive_scan")):
        return ""

    min_score = max(0.01, float(config.get("character_min_score", 0.35)))
    margin = max(0.0, float(config.get("character_margin", 0.08)))
    max_multi = max(2, int(config.get("character_max_multi", 4)))
    characters: list[tuple[str, float]] = []
    if preds and wd_tag_categories:
        for raw_tag, score in preds.items():
            category = wd_tag_categories.get(raw_tag, -1)
            if category in (4, "character") and score >= min_score:
                display = display_character_name(raw_tag)
                if display:
                    characters.append((display, float(score)))

    deduped: dict[str, float] = {}
    for name, score in characters:
        deduped[name] = max(deduped.get(name, 0.0), score)
    characters = sorted(deduped.items(), key=lambda item: item[1], reverse=True)

    if len(characters) >= 2 and config.get("character_create_multiple"):
        selected = characters[:max_multi]
        return safe_filename(" ".join(name for name, _score in selected))

    if characters:
        top_name, top_score = characters[0]
        second_score = characters[1][1] if len(characters) > 1 else 0.0
        if len(characters) == 1 or (top_score - second_score) >= margin:
            return top_name

    return source_character_hint(path, config.get("source_dir", ""), bool(config.get("recursive_scan")))


def prepend_character_folder(
    path: Path,
    sub: str,
    preds: dict[str, float] | None,
    config: dict[str, Any],
    wd_tag_categories: dict[str, Any] | None,
) -> str:
    """Prepend character subfolder to destination path if character detected."""
    character_folder = character_folder_for_image(path, preds, config, wd_tag_categories)
    if not character_folder:
        return sub
    sanitized_character = sanitize_subfolder_path(character_folder)
    sanitized_sub = sanitize_subfolder_path(sub)
    if not sanitized_character:
        return sub
    if sanitized_sub:
        first_part = Path(sanitized_sub).parts[0]
        if first_part.lower() == sanitized_character.lower():
            return sanitized_sub
        return str(Path(sanitized_character) / sanitized_sub)
    return sanitized_character


# ─────────────────────────────────────────────────────────────────────────────
# Cluster Naming & Optimization
# ─────────────────────────────────────────────────────────────────────────────

def semantic_name_similarities(
    names: list[str],
    siglip_embed_fn: Callable[[list[str]], torch.Tensor] | None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[tuple[int, int], float]:
    """Compute semantic pairwise similarities among candidate cluster folder names."""
    if len(names) < 2 or siglip_embed_fn is None:
        return {}
    try:
        with torch.inference_mode():
            embeddings = siglip_embed_fn(names).cpu()
        sims: dict[tuple[int, int], float] = {}
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                sims[(left, right)] = F.cosine_similarity(embeddings[left], embeddings[right], dim=0).item()
        return sims
    except Exception as exc:
        emit_log(log_callback, f"Warning: semantic folder-name comparison unavailable: {exc}")
        return {}


def optimize_cluster_folder_names(
    final_names: dict[int, str],
    cluster_candidate_tags: dict[int, list[tuple[str, float, float]]],
    config: dict[str, Any],
    siglip_embed_fn: Callable[[list[str]], torch.Tensor] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[int, str]:
    """Disambiguate cluster folder names using TF-IDF style tag rarity and semantic clustering."""
    if not final_names:
        return final_names

    generic_tokens = generic_folder_name_tokens()
    uniqueness = str(config.get("name_uniqueness_level", "Medium")).strip().lower()
    lexical_thresholds = {"low": 0.88, "medium": 0.74, "high": 0.62}
    semantic_thresholds = {"low": 0.94, "medium": 0.90, "high": 0.86}
    lexical_threshold = lexical_thresholds.get(uniqueness, lexical_thresholds["medium"])
    semantic_threshold = semantic_thresholds.get(uniqueness, semantic_thresholds["medium"])

    tag_cluster_counts: dict[str, int] = {}
    for candidates in cluster_candidate_tags.values():
        seen = {tag for tag, _coverage, _avg_score in candidates}
        for tag in seen:
            tag_cluster_counts[tag] = tag_cluster_counts.get(tag, 0) + 1
    total_clusters = max(len(final_names), 1)

    def is_generic_tag(tag: str) -> bool:
        tokens = folder_name_tokens(tag)
        return tag in generic_tokens or (bool(tokens) and tokens <= generic_tokens)

    def distinctive_tags(cluster_id: int) -> list[str]:
        scored: list[tuple[float, str]] = []
        for tag, coverage, avg_score in cluster_candidate_tags.get(cluster_id, []):
            if contains_year_token(tag) or is_generic_tag(tag):
                continue
            rarity = math.log((total_clusters + 1) / (1 + tag_cluster_counts.get(tag, 0))) + 1.0
            scored.append((coverage * avg_score * rarity, tag))
        scored.sort(reverse=True)
        return [tag for _score, tag in scored]

    optimized: dict[int, str] = {}
    for cluster_id, raw_name in final_names.items():
        name = safe_filename(raw_name)
        if config.get("canonical_folder_names"):
            name = canonical_folder_name(name)

        name_tokens = folder_name_tokens(name)
        non_generic = name_tokens - generic_tokens
        if not non_generic:
            distinct = distinctive_tags(cluster_id)
            if distinct:
                name = safe_filename(display_tag(distinct[0]))
        optimized[cluster_id] = name

    if not config.get("global_name_optimization", True):
        if config.get("hierarchical_folder_names"):
            optimized = {cluster_id: hierarchical_folder_name(name) for cluster_id, name in optimized.items()}
        return optimized

    cluster_ids = list(optimized.keys())
    names = [optimized[cluster_id] for cluster_id in cluster_ids]
    semantic_sims = semantic_name_similarities(names, siglip_embed_fn, log_callback)
    used_names: dict[str, int] = {}
    changed = 0

    for position, cluster_id in enumerate(cluster_ids):
        name = optimized[cluster_id]
        distinct = distinctive_tags(cluster_id)

        if name in used_names:
            for tag in distinct:
                display = display_tag(tag)
                candidate = safe_filename(display)[:60]
                if candidate not in used_names:
                    name = candidate
                    changed += 1
                    break
                candidate = safe_filename(f"{name} {display}")[:60]
                if candidate not in used_names:
                    name = candidate
                    changed += 1
                    break

        for previous_position in range(position):
            previous_id = cluster_ids[previous_position]
            previous_name = optimized[previous_id]
            lexical_sim = folder_name_similarity(name, previous_name)
            semantic_sim = semantic_sims.get((previous_position, position), semantic_sims.get((position, previous_position), 0.0))
            if lexical_sim < lexical_threshold and semantic_sim < semantic_threshold:
                continue

            clarified = False
            for tag in distinct:
                display = display_tag(tag)
                for candidate in (
                    safe_filename(display)[:60],
                    safe_filename(f"{name} {display}")[:60],
                ):
                    if (
                        candidate
                        and candidate not in used_names
                        and folder_name_similarity(candidate, previous_name) < lexical_threshold
                    ):
                        name = candidate
                        clarified = True
                        changed += 1
                        break
                if clarified:
                    break
            if not clarified:
                name = previous_name
                changed += 1
            break

        optimized[cluster_id] = name
        used_names[name] = cluster_id

    if config.get("hierarchical_folder_names"):
        optimized = {cluster_id: hierarchical_folder_name(name) for cluster_id, name in optimized.items()}

    if changed:
        emit_log(log_callback, f"Optimized folder names: adjusted/merged {changed} cluster names (uniqueness={uniqueness.title()}).")
    return optimized


# ─────────────────────────────────────────────────────────────────────────────
# CLI Tag Extraction (extract_tags.py implementation)
# ─────────────────────────────────────────────────────────────────────────────

def run_extract_tags_cli(argv: list[str] | None = None) -> int:
    """Standalone CLI utility to build metadata.json with the WD EVA02 or Camie tagger."""
    parser = argparse.ArgumentParser(description="Generate metadata.json with WD or Camie tagger")
    parser.add_argument("--data_dir", required=True, help="Folder containing your image collection")
    parser.add_argument("--engine", type=str, choices=["wd", "camie"], default="camie", help="Tagger engine: wd or camie")
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum tag confidence (0.0–1.0)")
    parser.add_argument("--exclude_folder", type=str, action="append", help="Folder name to exclude (repeatable)")
    parser.add_argument("--save_txt", action="store_true", help="Also write a .txt tag file next to each image")
    parser.add_argument("--lmdb", choices=["auto", "on", "off"], default="auto", help="LMDB image cache mode")
    parser.add_argument("--lmdb_dir", default=str(configured_lmdb_dir()), help="LMDB directory")
    parser.add_argument("--rebuild_lmdb", action="store_true", help="Force LMDB rebuild before tagging")
    parser.add_argument("--no_train_lmdb", action="store_true", help="Do not write train_dinov-compatible img:N keys")
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: folder not found: {data_dir}")
        return 1

    from .models import check_and_download_all
    check_and_download_all()

    paths = get_paths()
    print(f"Loading tag list ({args.engine})...")
    if args.engine == "camie":
        camie_dir = paths.camie_tagger_path
        camie_dir.mkdir(parents=True, exist_ok=True)
        model_path = camie_dir / "camie-tagger-v2.onnx"
        tags_path = camie_dir / "camie-tagger-v2-metadata.json"
        if not model_path.exists() or not tags_path.exists():
            print("Camie-Tagger-v2 model files are missing!")
            return 1

        with open(tags_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idx_to_tag = meta["dataset_info"]["tag_mapping"]["idx_to_tag"]
        tag_to_cat = meta["dataset_info"]["tag_mapping"]["tag_to_category"]
        max_idx = max([int(k) for k in idx_to_tag.keys()])
        wd_tags = [idx_to_tag.get(str(i), f"unknown_{i}") for i in range(max_idx + 1)]
        wd_tag_categories = tag_to_cat
    else:
        wd_dir = paths.wd_tagger_path
        model_path = wd_dir / "model.onnx"
        tags_path = wd_dir / "selected_tags.csv"
        if not model_path.exists() or not tags_path.exists():
            print(f"Error: WD Tagger model not found under {wd_dir}")
            return 1

        import pandas as pd
        df = pd.read_csv(tags_path)
        wd_tags = df["name"].tolist()
        wd_tag_categories = dict(zip(df["name"], df["category"]))

    import onnxruntime as ort
    print(f"Loading ONNX tagger from {model_path}...")
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("  [GPU] CUDAExecutionProvider available. Using GPU...")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        print("  [CPU] onnxruntime-gpu not available or CUDA unavailable. Using CPU...")
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    except Exception as e:
        print(f"  [Warning] Session init error ({e}). Forcing CPU...")
        session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    print(f"  Active provider: {session.get_providers()[0]}")

    print(f"Scanning folder {data_dir}...")
    image_paths = []
    exclude_folders = args.exclude_folder or []
    for root, dirs, files in os.walk(data_dir):
        if exclude_folders:
            dirs[:] = [d for d in dirs if d not in exclude_folders]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTS:
                image_paths.append(os.path.join(root, f))
    image_paths.sort(key=lambda p: clean_rel_path(os.path.relpath(p, start=data_dir)).lower())

    if not image_paths:
        print("No supported images found in that folder.")
        return 1

    print(f"Found {len(image_paths)} images. Starting tagging...")
    lmdb_env = prepare_extract_lmdb(
        image_paths=image_paths,
        data_dir=data_dir,
        lmdb_dir=args.lmdb_dir,
        mode=args.lmdb,
        rebuild=args.rebuild_lmdb,
    )

    metadata: dict[str, str] = {}
    tagged_samples: list[dict[str, str]] = []
    errors = 0
    t0 = time.time()
    try:
        from tqdm import tqdm
        iterator = tqdm(image_paths, desc="Tagging")
    except ImportError:
        iterator = image_paths

    try:
        for img_path in iterator:
            rel_path = clean_rel_path(os.path.relpath(img_path, start=data_dir))
            try:
                image_bytes = read_lmdb_image(lmdb_env, rel_path)
                if image_bytes is not None:
                    with Image.open(io.BytesIO(image_bytes)) as pil_img:
                        img_tensor = prepare_image(pil_img, engine=args.engine)
                else:
                    with Image.open(img_path) as pil_img:
                        img_tensor = prepare_image(pil_img, engine=args.engine)

                outputs = session.run(None, {input_name: img_tensor})
                if args.engine == "camie":
                    logits = outputs[1][0] if len(outputs) >= 2 else outputs[0][0]
                    preds = 1.0 / (1.0 + np.exp(-logits))
                else:
                    preds = outputs[0][0]

                general_tags: list[str] = []
                character_tags: list[str] = []
                for score, tag_name in zip(preds, wd_tags):
                    if score > args.threshold:
                        if tag_name in {"no_humans", "text_focus"} or tag_name.startswith("year_"):
                            continue
                        category = wd_tag_categories.get(tag_name, -1)
                        if args.engine == "camie":
                            is_char = category == "character"
                            is_rating = category in ["rating", "meta", "object"]
                        else:
                            is_char = category == 4
                            is_rating = category == 9

                        if is_rating:
                            continue
                        elif is_char:
                            character_tags.append(tag_name)
                        else:
                            general_tags.append(tag_name)

                final_tags = general_tags + character_tags
                tag_string = ", ".join(final_tags)
                metadata[rel_path] = tag_string
                tagged_samples.append({"image_path": os.path.abspath(img_path), "description": tag_string})

                if args.save_txt:
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(tag_string)
            except Exception as e:
                errors += 1
                print(f"\nError processing {img_path}: {e}")
    finally:
        if lmdb_env is not None:
            lmdb_env.close()

    out_file = os.path.join(data_dir, "metadata.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if not args.no_train_lmdb:
        try:
            write_train_lmdb(tagged_samples, data_dir, args.lmdb_dir, args.lmdb)
        except Exception as exc:
            if args.lmdb == "on":
                raise
            print(f"[LMDB] Could not prepare train cache: {exc}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Tagged: {len(metadata)}, errors: {errors}")
    print(f"Wrote metadata to: {out_file}")
    return 0


def main() -> None:
    sys.exit(run_extract_tags_cli())


if __name__ == "__main__":
    main()
