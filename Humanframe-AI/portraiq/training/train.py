
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.modeling import build_composition_model
from training.augment import build_eval_transform, build_train_transform
from training.dataset import (
    CompositionDataset,
    build_weighted_sampler,
    load_annotation_records,
    split_records,
)
from training.evaluate import evaluate_model
from training.losses import build_loss
from utils.gpu_config import create_grad_scaler, maybe_autocast, prepare_model
from utils.logger import create_logger


def _build_optimizer(model, config: Dict):
    base_lr = float(config["training"]["learning_rate"])
    wd = float(config["training"].get("weight_decay", 1e-5))
    lr_scale = float(config["model"].get("backbone_lr_scale", 0.1))

    params = [
        {"params": model.backbone.parameters(), "lr": base_lr * lr_scale},
        {"params": model.scorer_head.parameters(), "lr": base_lr},
    ]
    return torch.optim.AdamW(params, lr=base_lr, weight_decay=wd)


def _build_dataloaders(config: Dict):
    data_dir = Path("data")
    image_root = data_dir / "processed"
    anno_root = data_dir / "annotations"

    data_cfg = config.get("data", {})
    score_std_max = data_cfg.get("score_std_max")
    label_smoothing_alpha = float(data_cfg.get("label_smoothing_alpha", 0.0))
    records = load_annotation_records(
        anno_root,
        score_std_max=float(score_std_max) if score_std_max is not None else None,
        label_smoothing_alpha=label_smoothing_alpha,
    )
    train_r, val_r, test_r = split_records(
        records,
        train_ratio=float(config["data"]["train_split"]),
        val_ratio=float(config["data"]["val_split"]),
        test_ratio=float(config["data"]["test_split"]),
        seed=int(config["training"].get("seed", 42)),
    )

    image_size = int(config["data"]["image_size"])
    backbone_name = config["model"]["backbone"]
    train_ds = CompositionDataset(
        train_r,
        image_root=image_root,
        transform=build_train_transform(image_size=image_size, backbone_name=backbone_name),
    )
    val_ds = CompositionDataset(
        val_r,
        image_root=image_root,
        transform=build_eval_transform(image_size=image_size, backbone_name=backbone_name),
    )
    test_ds = CompositionDataset(
        test_r,
        image_root=image_root,
        transform=build_eval_transform(image_size=image_size, backbone_name=backbone_name),
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "No training samples found. Add annotation JSON files under data/annotations "
            "and ensure records are assigned to split='train' (or valid split ratios are set)."
        )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 4))
    use_weighted_sampling = bool(config["data"].get("weighted_sampling", False))
    train_sampler = build_weighted_sampler(train_r) if use_weighted_sampling else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def _set_backbone_trainable(model, trainable: bool):
    for param in model.backbone.parameters():
        param.requires_grad = trainable


def _build_scheduler(optimizer, config: Dict):
    scheduler_name = str(config["training"].get("lr_scheduler", "none")).lower()
    if scheduler_name == "cosine":
        epochs = int(config["training"]["epochs"])
        lr_min = float(config["training"].get("lr_min", 1.0e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, epochs),
            eta_min=lr_min,
        )
    return None


def _save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    best_metric: float,
    best_epoch: int = -1,
    scheduler=None,
    logger=None,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_mae": best_metric,
            "best_epoch": best_epoch,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        checkpoint_path,
    )
    size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    if logger is not None:
        logger.info("Checkpoint size: %.1f MB (%s)", size_mb, checkpoint_path.as_posix())
        if size_mb > 200.0:
            logger.warning(
                "Checkpoint size %.1fMB exceeds 200MB Pi deployment limit (%s)",
                size_mb,
                checkpoint_path.as_posix(),
            )


def run_training(config: Dict, resume: Optional[str] = None) -> Dict[str, float]:
    logger = create_logger("train")
    device, model = _prepare_model_and_device(config)
    train_loader, val_loader, test_loader = _build_dataloaders(config)

    criterion = build_loss(
        config["training"].get("loss", "smooth_l1"),
        variance_penalty=float(config["training"].get("variance_penalty", 0.0)),
    )
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    scaler = create_grad_scaler(config, device)
    grad_accum_steps = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    freeze_epochs = max(0, int(config["training"].get("freeze_epochs", 0)))

    start_epoch = 0
    best_val_mae = float("inf")
    best_epoch = -1
    patience = int(config["training"].get("early_stopping_patience", 7))
    no_improve_epochs = 0

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        if scheduler is not None:
            sched_state = ckpt.get("scheduler")
            if sched_state is not None:
                scheduler.load_state_dict(sched_state)
            else:
                for _ in range(start_epoch):
                    scheduler.step()
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    epochs = int(config["training"]["epochs"])
    checkpoint_dir = Path(config["training"].get("checkpoint_dir", "models"))
    last_ckpt_name = str(config["training"].get("last_checkpoint_name", "last.pth"))
    best_ckpt_name = str(config["training"].get("best_checkpoint_name", "best.pth"))

    # Two-phase schedule:
    # Phase 1: freeze backbone for first `freeze_epochs` epochs.
    # Phase 2: unfreeze and fine-tune full model.
    backbone_unfrozen_logged = False
    if freeze_epochs > 0 and start_epoch < freeze_epochs:
        _set_backbone_trainable(model, trainable=False)
        logger.info(
            "Epoch %d | Phase 1 active: backbone frozen (freeze_epochs=%d).",
            start_epoch + 1,
            freeze_epochs,
        )
    else:
        _set_backbone_trainable(model, trainable=True)
        if freeze_epochs > 0 and start_epoch >= freeze_epochs:
            backbone_unfrozen_logged = True
            logger.info(
                "Epoch %d | Backbone unfrozen — entering Phase 2 fine-tuning",
                start_epoch + 1,
            )

    for epoch in range(start_epoch, epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            _set_backbone_trainable(model, trainable=True)
            if not backbone_unfrozen_logged:
                logger.info(
                    "Epoch %d | Backbone unfrozen — entering Phase 2 fine-tuning",
                    epoch + 1,
                )
                backbone_unfrozen_logged = True

        model.train() # training mode of Pytorch modules
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            scores = batch["score"].to(device=device, dtype=torch.float32, non_blocking=True)

            with maybe_autocast(config, device):
                preds = model(images)
                loss = criterion(preds, scores)
                loss_for_backward = loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                accum=f"{grad_accum_steps}",
            )

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        logger.info(
            "Epoch %d | train_loss=%.4f val_mae=%.4f val_rmse=%.4f lr_head=%.8f lr_backbone=%.8f",
            epoch + 1,
            train_loss,
            val_metrics["mae"],
            val_metrics["rmse"],
            current_lrs[1] if len(current_lrs) > 1 else current_lrs[0],
            current_lrs[0],
        )
        logger.info(
            "Epoch %d | pred_score_min=%.4f pred_score_mean=%.4f pred_score_max=%.4f pred_score_std=%.4f",
            epoch + 1,
            float(val_metrics.get("pred_min", 0.0)),
            float(val_metrics.get("pred_mean", 0.0)),
            float(val_metrics.get("pred_max", 0.0)),
            float(val_metrics.get("pred_std", 0.0)),
        )

        _save_checkpoint(
            checkpoint_dir / last_ckpt_name,
            model,
            optimizer,
            epoch,
            best_val_mae,
            best_epoch=best_epoch,
            scheduler=scheduler,
            logger=logger,
        )

        in_phase1 = freeze_epochs > 0 and epoch < freeze_epochs
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch + 1
            no_improve_epochs = 0
            _save_checkpoint(
                checkpoint_dir / best_ckpt_name,
                model,
                optimizer,
                epoch,
                best_val_mae,
                best_epoch=best_epoch,
                scheduler=scheduler,
                logger=logger,
            )
            logger.info("New best checkpoint saved (val_mae=%.4f)", best_val_mae)
        else:
            if in_phase1:
                # Phase 1 is intentionally excluded from early-stopping countdown.
                no_improve_epochs = 0
                logger.info(
                    "Epoch %d | Phase 1 (backbone frozen) — early stopping paused until Phase 2",
                    epoch + 1,
                )
            else:
                no_improve_epochs += 1
                logger.info(
                    "No val_mae improvement for %d/%d epoch(s).",
                    no_improve_epochs,
                    patience,
                )
                if no_improve_epochs >= patience:
                    logger.info(
                        "Early stopping triggered at epoch %d. Best epoch: %d (val_mae=%.4f).",
                        epoch + 1,
                        best_epoch if best_epoch > 0 else (start_epoch + 1),
                        best_val_mae,
                    )
                    break

    test_metrics = evaluate_model(model, test_loader, device)
    if best_epoch > 0:
        logger.info("Best checkpoint remained at epoch %d with val_mae=%.4f", best_epoch, best_val_mae)
    logger.info("Test metrics: %s", test_metrics)
    return test_metrics


def run_evaluation(config: Dict, checkpoint: str) -> Dict[str, float]:
    logger = create_logger("eval")
    device, model = _prepare_model_and_device(config)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    _, _, test_loader = _build_dataloaders(config)
    metrics = evaluate_model(model, test_loader, device)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def _prepare_model_and_device(config: Dict):
    device_cfg = config.get("gpu", {}) # the config should have a "gpu" section
    use_cuda = device_cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda and device_cfg.get("visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_cfg["visible_devices"])

    model, _ = build_composition_model(config)
    model = prepare_model(model, config, device)
    return device, model
