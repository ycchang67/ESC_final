#!/usr/bin/env python3
"""Download FFHQ, apply person filtering, and build Level-3 annotations.

Pipeline:
1) Download FFHQ image files from Hugging Face Hub into data/raw/level3_good/ffhq/.
2) Run YOLO person filtering via portraiq/utils/pose_utils.py.
3) Copy passing images into data/processed/level3_good/.
4) Save annotations to data/annotations/level3_ffhq.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/build FFHQ Level-3 dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="marcosv/ffhq-dataset",
        help="HF dataset repo id (default: marcosv/ffhq-dataset)",
    )
    parser.add_argument("--max_images", type=int, default=70000, help="Maximum annotations to produce")
    parser.add_argument("--person_conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--person_model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded images and keep annotations")
    parser.add_argument("--timeout", type=int, default=30, help="HF download timeout in seconds")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data root (default: <repo>/portraiq/data)",
    )
    return parser.parse_args()


def repo_id_candidates(repo_id: str) -> List[str]:
    rid = repo_id.strip()
    candidates: List[str] = []
    if rid:
        candidates.append(rid)
    if rid.startswith("datasets/"):
        stripped = rid[len("datasets/") :]
        if stripped and stripped not in candidates:
            candidates.append(stripped)
    fallback_candidates = [
        "marcosv/ffhq-dataset",
        "student/FFHQ",
        "bitmind/ffhq",
    ]
    for item in fallback_candidates:
        if item not in candidates:
            candidates.append(item)
    return candidates


def repo_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "portraiq" / "data"


def ensure_dirs(data_root: Path) -> Dict[str, Path]:
    raw_dir = data_root / "raw" / "level3_good" / "ffhq"
    proc_dir = data_root / "processed" / "level3_good"
    ann_dir = data_root / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "processed": proc_dir, "annotations": ann_dir}


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def image_ext(path: str) -> str:
    return Path(path).suffix.lower()


def is_image_file(path: str) -> bool:
    return image_ext(path) in {".jpg", ".jpeg", ".png"}


def ffhq_image_id_from_path(path_str: str) -> str:
    stem = Path(path_str).stem
    match = re.search(r"(\d+)$", stem)
    if match:
        return f"ffhq_{match.group(1).zfill(max(5, len(match.group(1))))}"
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_")
    return f"ffhq_{safe or 'image'}"


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit(
            "Missing package huggingface_hub. Install first: python -m pip install huggingface_hub"
        )

    data_root = Path(args.data_root) if args.data_root else repo_data_root()
    paths = ensure_dirs(data_root)
    output_json = paths["annotations"] / "level3_ffhq.json"

    annotations: List[Dict] = []
    annotated_ids: Set[str] = set()
    if args.resume and output_json.exists():
        prev = load_json(output_json)
        if isinstance(prev, list):
            for item in prev:
                if isinstance(item, dict):
                    annotations.append(item)
                    image_id = str(item.get("image_id", "")).strip()
                    if image_id:
                        annotated_ids.add(image_id)

    # Import project detector after path setup.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    portraiq_root = repo_root / "portraiq"
    if str(portraiq_root) not in sys.path:
        sys.path.insert(0, str(portraiq_root))

    from utils.pose_utils import PersonDetector

    detector = PersonDetector(model_name=args.person_model, use_yolo=True)
    if detector.model is None:
        raise SystemExit("YOLO detector not available. Install ultralytics or fix YOLO model path.")

    api = HfApi()
    repo_id = None
    repo_files = None
    for candidate in repo_id_candidates(args.repo_id):
        try:
            repo_files = api.list_repo_files(repo_id=candidate, repo_type="dataset")
            repo_id = candidate
            break
        except Exception:
            continue
    if repo_id is None or repo_files is None:
        raise SystemExit(
            f"Failed to list files for dataset repo '{args.repo_id}'. "
            "Check the repo id and your network/auth setup."
        )

    image_files = [p for p in repo_files if is_image_file(p)]
    image_files.sort()

    if not image_files:
        raise SystemExit(
            f"No image files found in dataset repo '{repo_id}'. "
            "Check the repo id and file layout."
        )

    new_downloaded = 0
    new_kept = 0

    progress = tqdm(image_files, desc="FFHQ download/filter")
    for remote_path in progress:
        if len(annotations) >= int(args.max_images):
            break

        raw_name = Path(remote_path).name
        raw_path = paths["raw"] / raw_name
        image_id = ffhq_image_id_from_path(remote_path)

        if args.resume and image_id in annotated_ids:
            continue
        if image_id in annotated_ids:
            # Avoid duplicate annotations even when resume is disabled.
            continue

        if not raw_path.exists():
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    etag_timeout=int(args.timeout),
                )
            except Exception:
                continue
            shutil.copy2(downloaded, raw_path)
            new_downloaded += 1

        try:
            with Image.open(raw_path) as im:
                image = im.convert("RGB")
        except (UnidentifiedImageError, OSError):
            raw_path.unlink(missing_ok=True)
            continue

        det = detector.detect_largest_person(image_rgb=np.array(image))
        if det.source != "yolo" or float(det.confidence) < float(args.person_conf):
            continue

        proc_ext = raw_path.suffix.lower() or ".png"
        proc_name = f"{image_id}{proc_ext}"

        proc_path = paths["processed"] / proc_name
        shutil.copy2(raw_path, proc_path)

        annotations.append(
            {
                "image_id": image_id,
                "filename": f"level3_good/{proc_name}",
                "category": "portrait",
                "score": 8.0,
                "score_std": 0.0,
                "annotator": "ffhq_fixed",
                "split": "train",
            }
        )
        annotated_ids.add(image_id)
        new_kept += 1

        if new_kept % 200 == 0:
            dump_json(output_json, annotations)

    dump_json(output_json, annotations)
    print("FFHQ download complete")
    print(f"Repo: {repo_id}")
    print(f"Downloaded raw images (new): {new_downloaded}")
    print(f"Processed/kept images (new): {new_kept}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Output annotations: {output_json}")


if __name__ == "__main__":
    main()
