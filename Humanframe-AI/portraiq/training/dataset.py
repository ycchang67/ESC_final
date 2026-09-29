
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class CompositionDataset(Dataset):
    def __init__(self, records: Sequence[Dict], image_root: Path, transform=None) -> None:
        self.records = list(records)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.image_root / record["filename"]
        with Image.open(image_path) as pil:
            # Avoid PIL palette-transparency warning by converting P+transparency
            # images to RGBA before RGB.
            if pil.mode == "P" and "transparency" in pil.info:
                pil = pil.convert("RGBA")
            image = pil.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "score": float(record["score"]),
            "filename": record["filename"],
            "image_id": record["image_id"],
        }


def _parse_json_annotation(payload: Dict) -> Dict:
    if "filename" not in payload or "score" not in payload:
        raise ValueError("Each annotation must include 'filename' and 'score'.")

    score = float(payload["score"])
    if score < 0.0 or score > 10.0:
        raise ValueError("Annotation score must be in [0.0, 10.0].")

    return {
        "filename": payload["filename"],
        "score": score,
        "raw_score": float(payload.get("raw_score", score)),
        "score_std": float(payload["score_std"]) if payload.get("score_std") is not None else None,
        "split": payload.get("split", "unspecified"),
        "image_id": payload.get("image_id", Path(payload["filename"]).stem),
    }


def _apply_score_std_policy(
    record: Dict,
    score_std_max: float | None = None,
    label_smoothing_alpha: float = 0.0,
) -> Dict | None:
    if score_std_max is None:
        return record

    score_std = record.get("score_std")
    if score_std is None:
        return record

    if float(score_std) >= float(score_std_max):
        return None

    if 1.0 <= float(score_std) < float(score_std_max): # score std may not be used in this project
        alpha = min(max(float(label_smoothing_alpha), 0.0), 1.0)
        raw_score = float(record.get("raw_score", record["score"]))
        record["score"] = (raw_score * (1.0 - alpha)) + (5.0 * alpha)

    return record


def load_annotation_records(
    annotation_dir: Path,
    score_std_max: float | None = None,
    label_smoothing_alpha: float = 0.0,
) -> List[Dict]:
    annotation_dir = Path(annotation_dir)
    records: List[Dict] = []
    for path in sorted(annotation_dir.glob("*.json")):
        # Skip pipeline bookkeeping files that are not training labels.
        name = path.name.lower()
        if "skipped" in name or "raw_download" in name:
            continue

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            for item in payload:
                # Raw-manifest records should not be used for model training.
                if isinstance(item, dict) and item.get("storage") == "raw":
                    continue
                try:
                    parsed = _parse_json_annotation(item)
                    parsed = _apply_score_std_policy(
                        parsed,
                        score_std_max=score_std_max,
                        label_smoothing_alpha=label_smoothing_alpha,
                    )
                    if parsed is not None:
                        records.append(parsed)
                except ValueError:
                    continue
        else:
            if isinstance(payload, dict) and payload.get("storage") == "raw":
                continue
            try:
                parsed = _parse_json_annotation(payload)
                parsed = _apply_score_std_policy(
                    parsed,
                    score_std_max=score_std_max,
                    label_smoothing_alpha=label_smoothing_alpha,
                )
                if parsed is not None:
                    records.append(parsed)
            except ValueError:
                continue

    return records


def split_records(
    records: Sequence[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
    if not records:
        return [], [], []

    explicit_splits = {r["split"] for r in records if r["split"] in {"train", "val", "test"}}
    if explicit_splits:
        train = [r for r in records if r["split"] == "train"]
        val = [r for r in records if r["split"] == "val"]
        test = [r for r in records if r["split"] == "test"]
        return train, val, test

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test split ratios must sum to 1.0")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _quality_level_from_filename(filename: str) -> str:
    first = str(filename).split("/", 1)[0]
    if first in {"level1_poor", "level2_acceptable", "level3_good", "level4_excellent"}:
        return first
    return "other"


def build_weighted_sampler(records: Sequence[Dict]) -> WeightedRandomSampler:
    if not records:
        raise ValueError("Cannot build weighted sampler with empty records.")

    level_counts: Dict[str, int] = {}
    for rec in records:
        level = _quality_level_from_filename(rec.get("filename", ""))
        level_counts[level] = level_counts.get(level, 0) + 1

    weights: List[float] = []
    for rec in records:
        level = _quality_level_from_filename(rec.get("filename", ""))
        count = max(1, level_counts.get(level, 1))
        weights.append(1.0 / float(count))

    return WeightedRandomSampler(weights=weights, num_samples=len(records), replacement=True)
