
import torch
import torch.nn as nn


class SmoothL1WithVariancePenalty(nn.Module):
    def __init__(self, variance_penalty: float = 0.0, variance_target: float = 1.5) -> None:
        super().__init__()
        self.base = nn.SmoothL1Loss(beta=1.0)
        self.variance_penalty = float(variance_penalty)
        self.variance_target = float(variance_target)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smooth_l1_loss = self.base(predictions, targets)
        if self.variance_penalty <= 0.0:
            return smooth_l1_loss

        pred_std = predictions.std(unbiased=False)
        variance_loss = torch.relu(
            predictions.new_tensor(self.variance_target) - pred_std
        )
        return smooth_l1_loss + (self.variance_penalty * variance_loss) # let the model be corrected to identify the larger range of scores


def build_loss(name: str = "smooth_l1", variance_penalty: float = 0.0) -> nn.Module:
    key = name.lower()
    # Default to SmoothL1 (Huber) for robustness to noisy score labels.
    # Keep "mse" as a backward-compatible alias to avoid breaking old configs.
    if key in {"mse", "smooth_l1", "huber"}:
        return SmoothL1WithVariancePenalty(variance_penalty=variance_penalty, variance_target=1.5)
    raise ValueError("Unsupported loss. Use 'smooth_l1' (or alias 'mse').")
