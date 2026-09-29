
import torch
import torch.nn as nn


class CompositionModel(nn.Module):
    def __init__(self, backbone: nn.Module, scorer_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.scorer_head = scorer_head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.scorer_head(features)
