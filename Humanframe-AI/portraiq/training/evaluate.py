
from math import sqrt
from typing import Dict

import torch


def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            scores = batch["score"].to(device=device, dtype=torch.float32, non_blocking=True)
            outputs = model(images)
            outputs = torch.clamp(outputs, 0.0, 10.0)
            preds.extend(outputs.detach().cpu().tolist())
            targets.extend(scores.detach().cpu().tolist())

    if not preds:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "pred_min": 0.0,
            "pred_mean": 0.0,
            "pred_max": 0.0,
            "pred_std": 0.0,
        }

    abs_errors = [abs(p - t) for p, t in zip(preds, targets)]
    sq_errors = [(p - t) ** 2 for p, t in zip(preds, targets)]
    pred_min = min(preds)
    pred_max = max(preds)
    pred_mean = sum(preds) / len(preds)
    pred_var = sum((p - pred_mean) ** 2 for p in preds) / len(preds)

    return {
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": sqrt(sum(sq_errors) / len(sq_errors)),
        "pred_min": pred_min,
        "pred_mean": pred_mean,
        "pred_max": pred_max,
        "pred_std": sqrt(pred_var),
    }
