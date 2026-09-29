#!/usr/bin/env python3
"""Process Flickr raw Level-4 images into processed set + annotations.

Expected raw input:
  portraiq/data/raw/level4_excellent/flickr_*.jpg|png

Output:
  - portraiq/data/processed/level4_excellent/flickr_*.*
  - portraiq/data/annotations/level4_excellent.json (merged)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process Flickr raw images for Level-4 dataset")
    parser.add_argument("--person_conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--min_side", type=int, default=800, help="Minimum width/height to keep")
    parser.add_argument("--score", type=float, default=9.0, help="Fixed annotation score for Level-4")
    parser.add_argument("--save_every", type=int, default=200, help="Save JSON every N newly kept samples")
    parser.add_argument("--person_model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help=(
            "Directory containing downloaded Flickr files. "
            "Default: <data_root>/raw/level4_excellent"
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data root (default: <repo>/portraiq/data)",
    )
    return parser.parse_args()


def repo_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "portraiq" / "data"


def ensure_dirs(data_root: Path) -> Dict[str, Path]:
    raw_dir = data_root / "raw" / "level4_excellent"
    proc_dir = data_root / "processed" / "level4_excellent"
    ann_dir = data_root / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "processed": proc_dir, "annotations": ann_dir}


def deterministic_split(image_id: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> str:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    bucket = int(hashlib.md5(image_id.encode("utf-8")).hexdigest(), 16) % 1000
    train_cut = int(train_ratio * 1000)
    val_cut = train_cut + int(val_ratio * 1000)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def load_json(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def dump_json(path: Path, payload: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else repo_data_root()
    paths = ensure_dirs(data_root)
    output_json = paths["annotations"] / "level4_excellent.json"

    # Import detector from project package.
    repo_root = Path(__file__).resolve().parents[1]
    portraiq_root = repo_root / "portraiq"
    if str(portraiq_root) not in sys.path:
        sys.path.insert(0, str(portraiq_root))
    from utils.pose_utils import PersonDetector

    detector = PersonDetector(model_name=args.person_model, use_yolo=True)
    if detector.model is None:
        raise SystemExit("YOLO detector not available. Install ultralytics or fix YOLO model path.")

    annotations = load_json(output_json)
    annotated_ids = set()
    for item in annotations:
        if isinstance(item, dict):
            iid = str(item.get("image_id", "")).strip()
            if iid:
                annotated_ids.add(iid)

    source_dir = Path(args.input_dir) if args.input_dir else paths["raw"]
    raw_candidates = []
    if source_dir.exists():
        # Search recursively so gallery-dl nested group folders are supported.
        raw_candidates = [
            p
            for p in source_dir.rglob("*")
            if p.is_file() and p.name.lower().startswith("flickr_")
        ]
    raw_candidates.sort()

    scanned = 0
    kept = 0
    skipped_annotated = 0
    for raw_path in raw_candidates:
        scanned += 1
        image_id = raw_path.stem
        if image_id in annotated_ids:
            skipped_annotated += 1
            continue

        try:
            with Image.open(raw_path) as im:
                image = im.convert("RGB")
        except (UnidentifiedImageError, OSError):
            continue

        w, h = image.size
        if min(w, h) < int(args.min_side):
            continue

        det = detector.detect_largest_person(image_rgb=np.array(image))
        if det.source != "yolo" or float(det.confidence) < float(args.person_conf):
            continue

        proc_path = paths["processed"] / raw_path.name
        shutil.copy2(raw_path, proc_path)
        annotations.append(
            {
                "image_id": image_id,
                "filename": f"level4_excellent/{raw_path.name}",
                "category": "portrait",
                "score": float(args.score),
                "score_std": 0.0,
                "annotator": "flickr_fixed",
                "split": deterministic_split(image_id),
            }
        )
        annotated_ids.add(image_id)
        kept += 1

        if args.save_every > 0 and kept % int(args.save_every) == 0:
            dump_json(output_json, annotations)

    dump_json(output_json, annotations)
    print("Flickr processing complete")
    print(f"Scanned raw flickr images: {scanned}")
    print(f"Skipped (already annotated): {skipped_annotated}")
    print(f"New processed/annotated: {kept}")
    print(f"Output annotations: {output_json}")


if __name__ == "__main__":
    main()
