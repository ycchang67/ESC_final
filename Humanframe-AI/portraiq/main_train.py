
import argparse
from pathlib import Path

import yaml

from training.train import run_evaluation, run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Portraiq model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        metrics = run_training(config=config, resume=args.resume)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in evaluate mode")
        metrics = run_evaluation(config=config, checkpoint=args.checkpoint)

    print(metrics)


if __name__ == "__main__":
    main()
