
from contextlib import nullcontext

import torch


def prepare_model(model, config, device):
    model = model.to(device)
    multi_gpu = bool(config.get("training", {}).get("multi_gpu", False)) # for the multiple GPU environment
    if device.type == "cuda" and multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def maybe_autocast(config, device):
    mixed_precision = bool(config.get("training", {}).get("mixed_precision", True))
    if device.type == "cuda" and mixed_precision:
        return torch.cuda.amp.autocast()
    return nullcontext()


def create_grad_scaler(config, device):
    mixed_precision = bool(config.get("training", {}).get("mixed_precision", True))
    enabled = device.type == "cuda" and mixed_precision
    return torch.cuda.amp.GradScaler(enabled=enabled)
