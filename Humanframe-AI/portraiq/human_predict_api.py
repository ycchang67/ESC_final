import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from .inference.predict import InferenceEngine
except ImportError:  # pragma: no cover - script execution fallback
    from inference.predict import InferenceEngine


PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "accurate": {
        "config": "config.yaml",
        "checkpoint": "models/best.pth",
        "cpu_optimized": False,
    },
    "lightweight": {
        "config": "config_infer_rpi4.yaml",
        "checkpoint": "models/best_lite.pth",
        "cpu_optimized": True,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=8)
def _get_engine(config_path: str, checkpoint_path: str, cpu_optimized: bool) -> InferenceEngine:
    cfg = _load_config(Path(config_path))
    return InferenceEngine(
        config=cfg,
        checkpoint_path=checkpoint_path,
        cpu_optimized=cpu_optimized,
    )


def human_predict(
    image_path: str,
    profile: str = "accurate",
    checkpoint_path: str | None = None,
    config_path: str | None = None,
    cpu_optimized: bool | None = None,
    delete_below: float | None = None,
) -> Dict[str, Any]:
    profile_key = profile.lower()
    if profile_key not in PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported profile '{profile}'. Use one of: {', '.join(PROFILE_DEFAULTS)}")

    defaults = PROFILE_DEFAULTS[profile_key]
    root = _repo_root()
    resolved_config = Path(config_path) if config_path else (root / defaults["config"])
    resolved_checkpoint = (
        Path(checkpoint_path) if checkpoint_path else (root / defaults["checkpoint"])
    )
    resolved_cpu_opt = defaults["cpu_optimized"] if cpu_optimized is None else bool(cpu_optimized)

    if not resolved_config.exists():
        raise FileNotFoundError(f"Config not found: {resolved_config}")
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    engine = _get_engine(
        str(resolved_config.resolve()),
        str(resolved_checkpoint.resolve()),
        resolved_cpu_opt,
    )
    pred = engine.predict_score(str(image))
    score = float(pred["predicted_score"])

    deleted = False
    if delete_below is not None and score < float(delete_below):
        image.unlink(missing_ok=True)
        deleted = True

    return {
        "image_name": image.name,
        "image_path": str(image),
        "score": round(score, 4),
        "profile": profile_key,
        "deleted": deleted,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Simple API-like predictor for portrait aesthetic score")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--profile",
        default="accurate",
        choices=sorted(PROFILE_DEFAULTS.keys()),
        help="Model profile to use",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--cpu_optimized", action="store_true", help="Force CPU optimized mode")
    parser.add_argument(
        "--delete_below",
        type=float,
        default=None,
        help="If set, delete image when predicted score is below this threshold",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    result = human_predict(
        image_path=args.image,
        profile=args.profile,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        cpu_optimized=args.cpu_optimized if args.cpu_optimized else None,
        delete_below=args.delete_below,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# using image path to use the api call correctly
