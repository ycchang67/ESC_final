
import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Predicts raw portrait aesthetic score logits (unbounded during training)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Backbone wrappers already output pooled feature vectors. This head
        # expands capacity for better score calibration while staying lightweight.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)
