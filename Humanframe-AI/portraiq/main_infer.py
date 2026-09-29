
import argparse
import json
from pathlib import Path

import yaml

from inference.predict import InferenceEngine
from inference.visualize import save_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="Portraiq inference pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--cpu_optimized", action="store_true")
    parser.add_argument("--no_overlay", action="store_true")
    return parser.parse_args()


def load_config(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch_inference(
    input_dir: str,
    output_dir: str,
    config: dict,
    checkpoint_path: str,
    cpu_optimized: bool = False,
    save_overlay: bool = True,
):
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine(
        config=config,
        checkpoint_path=checkpoint_path,
        cpu_optimized=cpu_optimized,
    )

    results = []
    for image_path in sorted(in_dir.iterdir()):
        if image_path.suffix.lower() not in supported_ext:
            continue

        pred = engine.predict(str(image_path))
        if save_overlay:
            save_visualization(pred["overlay"], out_dir / image_path.name)
        pred.pop("overlay")
        results.append(pred)

    return results


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.image:
        engine = InferenceEngine(
            config=config,
            checkpoint_path=args.checkpoint,
            cpu_optimized=args.cpu_optimized,
        )
        pred = engine.predict(args.image)
        if not args.no_overlay:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            image_name = Path(args.image).name
            save_visualization(pred["overlay"], output_dir / image_name)
        pred.pop("overlay")
        print(json.dumps(pred, indent=2))
        return

    if args.input_dir:
        preds = run_batch_inference(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config,
            checkpoint_path=args.checkpoint,
            cpu_optimized=args.cpu_optimized,
            save_overlay=(not args.no_overlay),
        )
        print(json.dumps(preds, indent=2))
        return

    raise ValueError("Provide either --image or --input_dir")


if __name__ == "__main__":
    main()
