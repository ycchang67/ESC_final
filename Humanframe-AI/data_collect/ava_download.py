#!/usr/bin/env python3
"""AVA data pipeline for this project.

Recommended two-stage workflow:
1) download_raw: Download AVA images + scores into data/raw and write raw manifest JSON.
2) build_processed: Read raw manifest, run person filtering, copy kept samples to
   data/processed, and write training annotation JSON.

Legacy one-shot mode is available via --prepare-training in download_raw mode.
"""

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Download AVA and prepare project dataset")

    parser.add_argument(
        "--mode",
        type=str,
        default="download_raw",
        choices=("download_raw", "build_processed"),
        help="Pipeline stage to run",
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="trojblue/AVA-Huggingface",
        help="Hugging Face dataset id",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to project data directory (default: <repo>/portraiq/data)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=None,
        help="Path to annotations directory (default: <data-root>/annotations)",
    )

    parser.add_argument(
        "--raw-manifest-name",
        type=str,
        default="ava_raw_download.json",
        help="Filename for raw manifest JSON",
    )
    parser.add_argument(
        "--train-annotations-name",
        type=str,
        default="ava_from_hf.json",
        help="Filename for training annotation JSON",
    )
    parser.add_argument(
        "--skipped-log-name",
        type=str,
        default="ava_skipped_errors.json",
        help="Filename for skipped/errors log JSON",
    )

    parser.add_argument(
        "--poor-threshold",
        type=float,
        default=5.0,
        help="Scores < threshold go to level1_poor",
    )
    parser.add_argument(
        "--good-threshold",
        type=float,
        default=7.5,
        help="Scores >= threshold go to level3_good",
    )
    parser.add_argument(
        "--score-std-max",
        type=float,
        default=1.5,
        help="Discard samples with score_std >= this threshold.",
    )
    parser.add_argument(
        "--label-smoothing-alpha",
        type=float,
        default=0.1,
        help=(
            "For ambiguous samples (1.0 <= score_std < score_std_max), smooth score as "
            "raw_score*(1-alpha)+5.0*alpha."
        ),
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument(
        "--save-every",
        type=int,
        default=2000,
        help="Checkpoint manifests every N samples",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest files if they exist",
    )

    parser.add_argument(
        "--clean-levels",
        action="store_true",
        help=(
            "In download_raw mode: clear data/raw/level1..3 first. "
            "In build_processed mode: clear data/processed/level1..3 first."
        ),
    )
    parser.add_argument(
        "--clean-annotations",
        action="store_true",
        help="Delete output annotation JSONs for the selected mode before running",
    )

    parser.add_argument(
        "--person-filter",
        action="store_true",
        help="Enable person-only filtering (recommended for build_processed mode)",
    )
    parser.add_argument(
        "--person-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name for ultralytics backend",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.25,
        help="Confidence threshold for person detection",
    )
    parser.add_argument(
        "--person-imgsz",
        type=int,
        default=640,
        help="Detection size. For ultralytics: model imgsz. For torchvision/opencv: max long side after resize.",
    )
    parser.add_argument(
        "--person-device",
        type=str,
        default=None,
        help="Detector device override (e.g. 0, cpu)",
    )
    parser.add_argument(
        "--person-backend",
        type=str,
        default="auto",
        choices=("opencv", "auto", "ultralytics", "torchvision"),
        help=(
            "Person detector backend. "
            "'opencv' is fully local; 'auto' tries opencv, then ultralytics, then torchvision."
        ),
    )
    parser.add_argument(
        "--throttle-ms",
        type=int,
        default=0,
        help="Sleep this many milliseconds per sample in build_processed mode to reduce CPU load.",
    )

    parser.add_argument(
        "--prepare-training",
        action="store_true",
        help="Legacy one-shot behavior: after download_raw, also run build_processed",
    )

    return parser.parse_args()


def level_from_score(score: float, poor_threshold: float, good_threshold: float) -> str:
    if score < poor_threshold:
        return "level1_poor"
    if score >= good_threshold:
        return "level3_good"
    return "level2_acceptable"


def _extract_vote_distribution(row: Dict[str, Any]) -> Optional[List[float]]:
    list_keys = (
        "score_distribution",
        "vote_distribution",
        "votes",
        "ratings",
        "histogram",
    )
    for key in list_keys:
        val = row.get(key)
        if isinstance(val, (list, tuple)) and len(val) == 10:
            try:
                return [float(x) for x in val]
            except (TypeError, ValueError):
                continue

    patterns = (
        "vote_{}",
        "votes_{}",
        "score_{}",
        "rating_{}",
        "vote{}",
        "votes{}",
        "score{}",
        "rating{}",
    )
    votes: List[float] = []
    found_any = False
    for i in range(1, 11):
        current = None
        for pattern in patterns:
            key = pattern.format(i)
            if key in row:
                current = row[key]
                found_any = True
                break
        if current is None:
            votes = []
            break
        try:
            votes.append(float(current))
        except (TypeError, ValueError):
            votes = []
            break
    if found_any and len(votes) == 10:
        return votes

    return None


def _compute_score_std(votes: Optional[List[float]]) -> Optional[float]:
    if votes is None or len(votes) != 10:
        return None
    total_votes = float(sum(votes))
    if total_votes <= 0.0:
        return None

    scores = [float(i) for i in range(1, 11)]
    mean = sum(s * v for s, v in zip(scores, votes)) / total_votes
    var = sum(v * ((s - mean) ** 2) for s, v in zip(scores, votes)) / total_votes
    return math.sqrt(max(0.0, var))


def _apply_score_std_policy(
    raw_score: float,
    score_std: Optional[float],
    score_std_max: float,
    label_smoothing_alpha: float,
) -> Optional[float]:
    if score_std is None:
        return raw_score
    if score_std >= score_std_max:
        return None
    if 1.0 <= score_std < score_std_max:
        alpha = min(max(float(label_smoothing_alpha), 0.0), 1.0)
        return (raw_score * (1.0 - alpha)) + (5.0 * alpha)
    return raw_score


def deterministic_split(image_id: str, train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    bucket = int(hashlib.md5(str(image_id).encode("utf-8")).hexdigest(), 16) % 1000
    train_cut = int(train_ratio * 1000)
    val_cut = train_cut + int(val_ratio * 1000)

    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def ensure_level_dirs(root_dir: Path):
    for lv in ("level1_poor", "level2_acceptable", "level3_good"):
        (root_dir / lv).mkdir(parents=True, exist_ok=True)


def clean_level_dirs(root_dir: Path):
    for lv in ("level1_poor", "level2_acceptable", "level3_good"):
        target = root_dir / lv
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def init_person_detector(args):
    backend = args.person_backend

    if backend in ("auto", "opencv"):
        try:
            import cv2

            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            face_cascade = cv2.CascadeClassifier(
                str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            )
            if face_cascade.empty():
                raise RuntimeError("Failed to load OpenCV Haar cascade for face detection.")
            return {"backend": "opencv", "hog": hog, "face_cascade": face_cascade, "cv2": cv2}
        except Exception as exc:  # noqa: BLE001
            if backend == "opencv":
                raise SystemExit(
                    f"OpenCV backend failed to initialize: {exc}\\nInstall/verify opencv-python."
                ) from exc
            print(f"OpenCV backend unavailable ({exc}); trying ultralytics/torchvision fallback.")

    if backend in ("auto", "ultralytics"):
        try:
            from ultralytics import YOLO

            model = YOLO(args.person_model)
            return {"backend": "ultralytics", "model": model}
        except Exception as exc:  # noqa: BLE001
            if backend == "ultralytics":
                raise SystemExit(
                    f"Ultralytics backend failed to initialize: {exc}\\n"
                    "Try --person-backend torchvision."
                ) from exc
            print(f"Ultralytics unavailable ({exc}); falling back to torchvision detector.")

    if backend in ("auto", "torchvision"):
        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
            )
            from torchvision.transforms.functional import to_tensor

            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
            model.eval()

            device = "cpu"
            if args.person_device:
                device = str(args.person_device)
            elif torch.cuda.is_available():
                device = "cuda:0"
            model.to(device)

            return {"backend": "torchvision", "model": model, "device": device, "to_tensor": to_tensor}
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Torchvision backend failed to initialize: {exc}\\n"
                "Install/verify torchvision and torch."
            ) from exc

    raise SystemExit(f"Unknown person backend: {backend}")


def has_person(detector, image, conf: float, imgsz: int, device: str):
    backend = detector["backend"]
    detection_image = image

    if imgsz and imgsz > 0 and backend in ("opencv", "torchvision"):
        w, h = image.size
        long_side = max(w, h)
        if long_side > imgsz:
            scale = float(imgsz) / float(long_side)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            detection_image = image.resize((nw, nh))

    if backend == "opencv":
        import numpy as np

        cv2 = detector["cv2"]
        rgb = np.array(detection_image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        faces = detector["face_cascade"].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24)
        )
        if len(faces) > 0:
            return True

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rects, weights = detector["hog"].detectMultiScale(
            bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        if len(rects) == 0:
            return False

        threshold = max(0.0, float(conf))
        return any(float(w) >= threshold for w in weights)

    model = detector["model"]

    if backend == "ultralytics":
        kwargs = {
            "source": image,
            "verbose": False,
            "classes": [0],
            "conf": conf,
            "imgsz": imgsz,
        }
        if device:
            kwargs["device"] = device
        results = model.predict(**kwargs)
        if not results:
            return False
        boxes = results[0].boxes
        return boxes is not None and len(boxes) > 0

    if backend == "torchvision":
        import torch

        tensor = detector["to_tensor"](detection_image)
        with torch.inference_mode():
            pred = model([tensor.to(detector["device"])])[0]
        labels = pred["labels"].detach().cpu().tolist()
        scores = pred["scores"].detach().cpu().tolist()
        for label, score in zip(labels, scores):
            if int(label) == 1 and float(score) >= conf:
                return True
        return False

    raise RuntimeError(f"Unsupported detector backend: {backend}")


def run_download_raw(args, data_root: Path, annotations_dir: Path):
    try:
        from datasets import load_dataset
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Missing packages. Install first: python -m pip install --user datasets pillow tqdm"
        ) from exc

    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    raw_root = data_root / "raw"
    raw_manifest_path = annotations_dir / args.raw_manifest_name
    skipped_path = annotations_dir / args.skipped_log_name

    if args.clean_levels:
        clean_level_dirs(raw_root)
    else:
        ensure_level_dirs(raw_root)

    if args.clean_annotations:
        for p in (raw_manifest_path, skipped_path):
            if p.exists():
                p.unlink()

    ds = load_dataset(args.dataset_id, split=args.split)

    raw_records: List[Dict] = []
    skipped_records: List[Dict] = []
    done_ids = set()
    skipped_high_std = 0

    if args.resume:
        raw_records = load_json_list(raw_manifest_path)
        done_ids = {str(rec.get("image_id", "")) for rec in raw_records}
        skipped_records = load_json_list(skipped_path)
        print(f"Resume: loaded {len(raw_records)} raw records.")

    for idx in tqdm(range(len(ds)), desc="Downloading AVA -> raw"):
        try:
            row = ds[idx]
        except Exception as exc:  # noqa: BLE001
            skipped_records.append({"stage": "download_raw", "index": idx, "error": str(exc)})
            if args.save_every > 0 and (idx + 1) % args.save_every == 0:
                dump_json(raw_manifest_path, raw_records)
                dump_json(skipped_path, skipped_records)
            continue

        try:
            votes = _extract_vote_distribution(row)
            score_std = _compute_score_std(votes)
            raw_score = float(row["mean_score"])
            filtered_score = _apply_score_std_policy(
                raw_score=raw_score,
                score_std=score_std,
                score_std_max=float(args.score_std_max),
                label_smoothing_alpha=float(args.label_smoothing_alpha),
            )

            if filtered_score is None:
                skipped_high_std += 1
                skipped_records.append(
                    {
                        "stage": "download_raw",
                        "index": idx,
                        "image_id": str(row.get("image_id", "")),
                        "reason": "high_score_std",
                        "score_std": round(float(score_std), 4) if score_std is not None else None,
                        "score_std_max": float(args.score_std_max),
                    }
                )
                continue

            score = float(filtered_score)
            level = level_from_score(score, args.poor_threshold, args.good_threshold)
            image_id = str(row["image_id"])
            if image_id in done_ids:
                continue

            rel = Path(level) / f"{image_id}.jpg"
            raw_out = raw_root / rel
            raw_out.parent.mkdir(parents=True, exist_ok=True)

            image = row["image"]
            image.save(raw_out, quality=95)

            split = deterministic_split(image_id, args.train_ratio, args.val_ratio, args.test_ratio)
            raw_records.append(
                {
                    "image_id": image_id,
                    "filename": str(rel).replace("\\", "/"),
                    "score": round(score, 4),
                    "raw_score": round(raw_score, 4),
                    "score_std": round(float(score_std), 4) if score_std is not None else None,
                    "split": split,
                    "source": "AVA",
                    "level": level,
                    "storage": "raw",
                }
            )
            done_ids.add(image_id)
        except Exception as exc:  # noqa: BLE001
            skipped_records.append({"stage": "download_raw", "index": idx, "error": str(exc)})

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            dump_json(raw_manifest_path, raw_records)
            dump_json(skipped_path, skipped_records)

    dump_json(raw_manifest_path, raw_records)
    dump_json(skipped_path, skipped_records)

    print("Saved raw images:", len(raw_records))
    print("Raw manifest:", raw_manifest_path)
    print("Download skipped/corrupted:", len(skipped_records))
    print("Skipped (high score_std):", skipped_high_std)
    print("Skipped log:", skipped_path)


def run_build_processed(args, data_root: Path, annotations_dir: Path):
    try:
        from PIL import Image, ImageFile
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit(
            "Missing packages. Install first: python -m pip install --user pillow tqdm"
        ) from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    raw_root = data_root / "raw"
    processed_root = data_root / "processed"

    raw_manifest_path = annotations_dir / args.raw_manifest_name
    train_ann_path = annotations_dir / args.train_annotations_name
    skipped_path = annotations_dir / args.skipped_log_name

    if not raw_manifest_path.exists():
        raise SystemExit(f"Raw manifest not found: {raw_manifest_path}. Run --mode download_raw first.")

    if args.clean_levels:
        clean_level_dirs(processed_root)
    else:
        ensure_level_dirs(processed_root)

    if args.clean_annotations:
        for p in (train_ann_path, skipped_path):
            if p.exists():
                p.unlink()

    raw_records = load_json_list(raw_manifest_path)
    if not raw_records:
        raise SystemExit(f"Raw manifest is empty: {raw_manifest_path}")

    detector = None
    if args.person_filter:
        detector = init_person_detector(args)
        print(
            f"Person filter enabled: backend={detector['backend']}, conf={args.person_conf}, imgsz={args.person_imgsz}"
        )
    else:
        print("Person filter disabled: all raw samples will be copied to processed.")

    train_records: List[Dict] = []
    skipped_records: List[Dict] = []
    done_ids = set()

    if args.resume:
        train_records = load_json_list(train_ann_path)
        done_ids = {str(rec.get("image_id", "")) for rec in train_records}
        skipped_records = load_json_list(skipped_path)
        print(f"Resume: loaded {len(train_records)} processed records.")

    skipped_non_person = 0
    skipped_high_std = 0

    for idx, rec in enumerate(tqdm(raw_records, desc="Filtering raw -> processed")):
        image_id = str(rec.get("image_id", ""))
        if image_id in done_ids:
            continue

        try:
            raw_score = float(rec.get("raw_score", rec["score"]))
            score_std = rec.get("score_std")
            score_std = float(score_std) if score_std is not None else None
            final_score = _apply_score_std_policy(
                raw_score=raw_score,
                score_std=score_std,
                score_std_max=float(args.score_std_max),
                label_smoothing_alpha=float(args.label_smoothing_alpha),
            )
            if final_score is None:
                skipped_high_std += 1
                skipped_records.append(
                    {
                        "stage": "build_processed",
                        "index": idx,
                        "image_id": image_id,
                        "filename": rec.get("filename", ""),
                        "reason": "high_score_std",
                        "score_std": round(float(score_std), 4) if score_std is not None else None,
                        "score_std_max": float(args.score_std_max),
                    }
                )
                continue

            rel = rec["filename"]
            raw_path = raw_root / rel
            if not raw_path.exists():
                skipped_records.append(
                    {
                        "stage": "build_processed",
                        "index": idx,
                        "image_id": image_id,
                        "filename": rel,
                        "reason": "missing_raw_file",
                    }
                )
                continue

            with Image.open(raw_path) as im:
                image = im.convert("RGB")

            if detector is not None and not has_person(
                detector, image, args.person_conf, args.person_imgsz, args.person_device
            ):
                skipped_non_person += 1
                skipped_records.append(
                    {
                        "stage": "build_processed",
                        "index": idx,
                        "image_id": image_id,
                        "filename": rel,
                        "reason": "no_person_detected",
                    }
                )
                continue

            proc_out = processed_root / rel
            proc_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(raw_path), str(proc_out))

            train_records.append(
                {
                    "image_id": image_id,
                    "filename": rel,
                    "score": float(final_score),
                    "raw_score": raw_score,
                    "score_std": round(float(score_std), 4) if score_std is not None else None,
                    "split": rec.get("split", "train"),
                }
            )
            done_ids.add(image_id)
        except Exception as exc:  # noqa: BLE001
            skipped_records.append(
                {
                    "stage": "build_processed",
                    "index": idx,
                    "image_id": image_id,
                    "filename": rec.get("filename", ""),
                    "error": str(exc),
                }
            )

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            dump_json(train_ann_path, train_records)
            dump_json(skipped_path, skipped_records)

        if args.throttle_ms > 0:
            time.sleep(args.throttle_ms / 1000.0)

    dump_json(train_ann_path, train_records)
    dump_json(skipped_path, skipped_records)

    print("Prepared processed images:", len(train_records))
    print("Training annotations:", train_ann_path)
    print("Skipped/corrupted samples:", len(skipped_records))
    print("Skipped (no person detected):", skipped_non_person)
    print("Skipped (high score_std):", skipped_high_std)
    print("Skipped log:", skipped_path)


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root) if args.data_root else (repo_root / "portraiq" / "data")
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else (data_root / "annotations")
    annotations_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "download_raw":
        run_download_raw(args, data_root, annotations_dir)
        if args.prepare_training:
            run_build_processed(args, data_root, annotations_dir)
        return

    if args.mode == "build_processed":
        run_build_processed(args, data_root, annotations_dir)
        return

    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
