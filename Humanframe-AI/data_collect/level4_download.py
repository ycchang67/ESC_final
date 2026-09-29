#!/usr/bin/env python3
"""Unified Level-4 downloader (Pexels + Unsplash) for excellent portraits.

Pipeline:
1) Download raw images into data/raw/level4_excellent.
2) Run YOLO person filtering via portraiq/utils/pose_utils.py.
3) Copy passing images into data/processed/level4_excellent.
4) Save unified annotations to data/annotations/level4_excellent.json.
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and build Level-4 excellent dataset")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=("pexels", "unsplash", "all"),
        help="Data source to use",
    )
    parser.add_argument("--max_images", type=int, default=5000, help="Maximum total annotated images")
    parser.add_argument(
        "--target_new_raw",
        type=int,
        default=0,
        help="Stop after downloading this many NEW raw images (0 = disabled)",
    )
    parser.add_argument(
        "--per_keyword",
        type=int,
        default=2000,
        help="Maximum number of candidates per keyword (Pexels/Unsplash)",
    )
    parser.add_argument("--person_conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output annotation JSON")

    parser.add_argument("--clean", action="store_true", help="Clean raw/processed level4 and output annotation")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--save_every", type=int, default=200, help="Save annotation every N newly kept images")
    parser.add_argument("--sleep_ms", type=int, default=0, help="Sleep milliseconds between requests")
    parser.add_argument("--min_side", type=int, default=384, help="Minimum accepted width/height")
    parser.add_argument("--person_model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument(
        "--use_pexels_curated",
        action="store_true",
        help="Also use Pexels curated endpoint for additional non-search candidates",
    )
    parser.add_argument(
        "--use_unsplash_popular",
        action="store_true",
        help="Also use Unsplash popular feed for additional non-search candidates",
    )

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

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


def dump_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deterministic_split(image_id: str, train_ratio: float, val_ratio: float, test_ratio: float) -> str:
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


def download_binary(url: str, dst: Path, timeout: int) -> bool:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dst.write_bytes(resp.content)
        return True
    except Exception:
        return False


def pexels_search(api_key: str, query: str, page: int, per_page: int, timeout: int) -> Dict:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "page": page, "per_page": per_page, "orientation": "portrait"}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def pexels_curated(api_key: str, page: int, per_page: int, timeout: int) -> Dict:
    url = "https://api.pexels.com/v1/curated"
    headers = {"Authorization": api_key}
    params = {"page": page, "per_page": per_page}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def iter_pexels_candidates(
    api_key: str,
    per_keyword: int,
    timeout: int,
) -> Iterable[Tuple[str, str]]:
    keywords = [
        "portrait",
        "street portrait",
        "environmental portrait",
        "editorial portrait",
        "studio portrait",
        "fashion portrait",
        "candid portrait",
        "black and white portrait",
        "fine art portrait",
        "dramatic portrait lighting",
    ]
    for kw in keywords:
        page = 1
        seen_for_kw = 0
        while seen_for_kw < per_keyword:
            per_page = min(80, max(1, per_keyword - seen_for_kw))
            try:
                payload = pexels_search(api_key, kw, page=page, per_page=per_page, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"Pexels search failed for '{kw}' page {page}: {exc}")
                break

            photos = payload.get("photos", []) if isinstance(payload, dict) else []
            if not photos:
                break

            for photo in photos:
                pid = str(photo.get("id", "")).strip()
                if not pid:
                    continue
                src = photo.get("src", {})
                image_url = src.get("large2x") or src.get("large") or src.get("original")
                if not image_url:
                    continue
                seen_for_kw += 1
                yield pid, str(image_url)
                if seen_for_kw >= per_keyword:
                    break

            page += 1


def iter_pexels_curated_candidates(
    api_key: str,
    cap: int,
    timeout: int,
) -> Iterable[Tuple[str, str]]:
    page = 1
    seen = 0
    while seen < cap:
        per_page = min(80, max(1, cap - seen))
        try:
            payload = pexels_curated(api_key, page=page, per_page=per_page, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"Pexels curated failed on page {page}: {exc}")
            break
        photos = payload.get("photos", []) if isinstance(payload, dict) else []
        if not photos:
            break
        for photo in photos:
            pid = str(photo.get("id", "")).strip()
            if not pid:
                continue
            src = photo.get("src", {})
            image_url = src.get("large2x") or src.get("large") or src.get("original")
            if not image_url:
                continue
            seen += 1
            yield pid, str(image_url)
            if seen >= cap:
                break
        page += 1


def unsplash_search(api_key: str, query: str, page: int, per_page: int, timeout: int) -> Dict:
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"query": query, "page": page, "per_page": per_page, "orientation": "portrait"}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def unsplash_popular(api_key: str, page: int, per_page: int, timeout: int) -> Dict:
    url = "https://api.unsplash.com/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"page": page, "per_page": per_page, "order_by": "popular"}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def iter_unsplash_candidates(
    api_key: str,
    per_keyword: int,
    timeout: int,
) -> Iterable[Tuple[str, str]]:
    keywords = [
        "portrait photography",
        "professional portrait",
        "street portrait",
        "environmental portrait",
        "editorial portrait",
        "studio portrait",
        "fashion portrait",
        "moody portrait",
        "cinematic portrait",
        "headshot portrait",
    ]
    for kw in keywords:
        page = 1
        seen_for_kw = 0
        while seen_for_kw < per_keyword:
            per_page = min(30, max(1, per_keyword - seen_for_kw))
            try:
                payload = unsplash_search(api_key, kw, page=page, per_page=per_page, timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"Unsplash search failed for '{kw}' page {page}: {exc}")
                break

            photos = payload.get("results", []) if isinstance(payload, dict) else []
            if not photos:
                break

            for photo in photos:
                pid = str(photo.get("id", "")).strip()
                if not pid:
                    continue
                width = int(photo.get("width") or 0)
                height = int(photo.get("height") or 0)
                if width < 800 or height < 800:
                    continue
                urls = photo.get("urls", {})
                image_url = urls.get("regular") or urls.get("full") or urls.get("raw")
                if not image_url:
                    continue
                seen_for_kw += 1
                yield pid, str(image_url)
                if seen_for_kw >= per_keyword:
                    break

            page += 1


def iter_unsplash_popular_candidates(
    api_key: str,
    cap: int,
    timeout: int,
) -> Iterable[Tuple[str, str]]:
    page = 1
    seen = 0
    while seen < cap:
        per_page = min(30, max(1, cap - seen))
        try:
            payload = unsplash_popular(api_key, page=page, per_page=per_page, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"Unsplash popular failed on page {page}: {exc}")
            break

        photos = payload if isinstance(payload, list) else []
        if not photos:
            break

        for photo in photos:
            pid = str(photo.get("id", "")).strip()
            if not pid:
                continue
            width = int(photo.get("width") or 0)
            height = int(photo.get("height") or 0)
            if width < 800 or height < 800:
                continue
            urls = photo.get("urls", {})
            image_url = urls.get("regular") or urls.get("full") or urls.get("raw")
            if not image_url:
                continue
            seen += 1
            yield pid, str(image_url)
            if seen >= cap:
                break
        page += 1


def chain_candidates(*iterables: Iterable[Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
    for it in iterables:
        for item in it:
            yield item


def main() -> None:
    args = parse_args()

    if args.source in ("pexels", "all"):
        pexels_api_key = os.getenv("PEXELS_API_KEY")
        if not pexels_api_key:
            raise SystemExit("PEXELS_API_KEY is required when --source is 'pexels' or 'all'.")
    else:
        pexels_api_key = None
    if args.source in ("unsplash", "all"):
        unsplash_api_key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not unsplash_api_key:
            raise SystemExit("UNSPLASH_ACCESS_KEY is required when --source is 'unsplash' or 'all'.")
    else:
        unsplash_api_key = None

    data_root = Path(args.data_root) if args.data_root else repo_data_root()
    paths = ensure_dirs(data_root)
    output_json = paths["annotations"] / "level4_excellent.json"
    legacy_jsons = [
        paths["annotations"] / "level4_pexels.json",
        paths["annotations"] / "level4_unsplash.json",
    ]

    if args.clean:
        for d in (paths["raw"], paths["processed"]):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        if output_json.exists():
            output_json.unlink()
        for legacy in legacy_jsons:
            if legacy.exists():
                legacy.unlink()

    annotations: List[Dict] = []
    annotated_ids = set()
    if args.resume and output_json.exists():
        prev = load_json(output_json)
        if isinstance(prev, list):
            annotations = prev
            for item in prev:
                if isinstance(item, dict):
                    image_id = str(item.get("image_id", "")).strip()
                    if image_id:
                        annotated_ids.add(image_id)
    elif args.resume:
        # One-time migration path from legacy split files.
        migrated = 0
        for legacy in legacy_jsons:
            prev = load_json(legacy)
            if not isinstance(prev, list):
                continue
            for item in prev:
                if not isinstance(item, dict):
                    continue
                image_id = str(item.get("image_id", "")).strip()
                if not image_id or image_id in annotated_ids:
                    continue
                annotations.append(item)
                annotated_ids.add(image_id)
                migrated += 1
        if migrated > 0:
            dump_json(output_json, annotations)
            print(f"Migrated {migrated} annotations into {output_json.name} from legacy level4 files.")

    # Import project detector only after path setup.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    portraiq_root = repo_root / "portraiq"
    if str(portraiq_root) not in sys.path:
        sys.path.insert(0, str(portraiq_root))

    from utils.pose_utils import PersonDetector

    detector = PersonDetector(model_name=args.person_model, use_yolo=True)
    if detector.model is None:
        raise SystemExit("YOLO detector not available. Install ultralytics or fix YOLO model path.")

    source_plan: List[Tuple[str, float, str, Iterable[Tuple[str, str]]]] = []
    if args.source in ("pexels", "all"):
        source_plan.append(
            (
                "pexels",
                9.5,
                "pexels_fixed",
                chain_candidates(
                    iter_pexels_candidates(
                        api_key=str(pexels_api_key),
                        per_keyword=int(args.per_keyword),
                        timeout=int(args.timeout),
                    ),
                    iter_pexels_curated_candidates(
                        api_key=str(pexels_api_key),
                        cap=int(args.per_keyword) if args.use_pexels_curated else 0,
                        timeout=int(args.timeout),
                    ) if args.use_pexels_curated else [],
                ),
            )
        )
    if args.source in ("unsplash", "all"):
        source_plan.append(
            (
                "unsplash",
                9.5,
                "unsplash_fixed",
                chain_candidates(
                    iter_unsplash_candidates(
                        api_key=str(unsplash_api_key),
                        per_keyword=int(args.per_keyword),
                        timeout=int(args.timeout),
                    ),
                    iter_unsplash_popular_candidates(
                        api_key=str(unsplash_api_key),
                        cap=int(args.per_keyword) if args.use_unsplash_popular else 0,
                        timeout=int(args.timeout),
                    ) if args.use_unsplash_popular else [],
                ),
            )
        )

    new_downloaded = 0
    new_kept = 0
    existing_raw_names = {p.name for p in paths["raw"].glob("*.jpg")}
    seen_image_ids = set()

    for source_name, fixed_score, annotator, candidates in source_plan:
        retrieved_from_api = 0
        for ext_id, image_url in candidates:
            retrieved_from_api += 1
            if len(annotations) >= int(args.max_images):
                break

            image_id = f"{source_name}_{ext_id}"
            if image_id in seen_image_ids:
                continue
            seen_image_ids.add(image_id)
            raw_name = f"{image_id}.jpg"
            raw_path = paths["raw"] / raw_name
            proc_path = paths["processed"] / raw_name

            if image_id in annotated_ids:
                continue

            # Resume mode should skip already-downloaded files when possible.
            if raw_name in existing_raw_names and not raw_path.exists():
                existing_raw_names.remove(raw_name)

            if not raw_path.exists():
                ok = download_binary(image_url, raw_path, timeout=int(args.timeout))
                if not ok:
                    continue
                new_downloaded += 1
                existing_raw_names.add(raw_name)
                if int(args.target_new_raw) > 0 and new_downloaded >= int(args.target_new_raw):
                    # We still run filtering on this image; stop after current loop step.
                    pass

            try:
                with Image.open(raw_path) as im:
                    image = im.convert("RGB")
            except (UnidentifiedImageError, OSError):
                raw_path.unlink(missing_ok=True)
                continue

            w, h = image.size
            if min(w, h) < int(args.min_side):
                continue

            det = detector.detect_largest_person(image_rgb=np.array(image))
            if det.source != "yolo" or float(det.confidence) < float(args.person_conf):
                continue

            shutil.copy2(raw_path, proc_path)
            split = deterministic_split(
                image_id=image_id,
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                test_ratio=float(args.test_ratio),
            )
            annotations.append(
                {
                    "image_id": image_id,
                    "filename": f"level4_excellent/{raw_name}",
                    "category": "portrait",
                    "score": float(fixed_score),
                    "annotator": annotator,
                    "split": split,
                }
            )
            annotated_ids.add(image_id)
            new_kept += 1

            if args.save_every > 0 and new_kept % args.save_every == 0:
                dump_json(output_json, annotations)

            if args.sleep_ms > 0:
                time.sleep(float(args.sleep_ms) / 1000.0)

            if int(args.target_new_raw) > 0 and new_downloaded >= int(args.target_new_raw):
                break

        print(
            f"DEBUG | source={source_name} | candidate_urls_retrieved_before_yolo={retrieved_from_api}"
        )
        if len(annotations) >= int(args.max_images):
            break
        if int(args.target_new_raw) > 0 and new_downloaded >= int(args.target_new_raw):
            break

    dump_json(output_json, annotations)
    print("Level4 download complete")
    print(f"Source mode: {args.source}")
    print(f"Downloaded raw images (new): {new_downloaded}")
    print(f"Processed/kept images (new): {new_kept}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Output annotations: {output_json}")


if __name__ == "__main__":
    main()
