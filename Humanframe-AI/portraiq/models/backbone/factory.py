
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

try:
    import open_clip
except ImportError:  # pragma: no cover - optional dependency at runtime
    open_clip = None

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

BACKBONE_DIMS = {
    "efficientnet_b4": 1792,
    "clip_vit_l14": 768,
}


class TorchvisionBackbone(nn.Module):
    def __init__(self, model: nn.Module, kind: str) -> None:
        super().__init__()
        self.model = model
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "efficientnet_b4":
            x = self.model.features(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        raise ValueError(f"Unsupported torchvision backbone kind: {self.kind}")


class OpenCLIPBackbone(nn.Module):
    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # open_clip.create_model_and_transforms(...).visual returns a visual tower
        # (e.g. VisionTransformer) which is callable but does not expose
        # encode_image(). Full CLIP models do expose encode_image().
        if hasattr(self.clip_model, "encode_image"):
            return self.clip_model.encode_image(x)
        return self.clip_model(x)


def _create_torchvision_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b4(weights=weights)
        return TorchvisionBackbone(net, kind=name), BACKBONE_DIMS[name]

    raise ValueError(f"Unknown torchvision backbone: {name}")


def _create_clip_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if open_clip is None:
        raise ImportError("open-clip-torch is required for CLIP backbones. Install requirements first.")

    if name == "clip_vit_l14":
        clip_model_name = "ViT-L-14"
    else:
        raise ValueError(f"Unsupported CLIP backbone: {name}")

    pretrained_tag = "openai" if pretrained else None
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained_tag)
    return OpenCLIPBackbone(clip_model.visual), BACKBONE_DIMS[name]


def create_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Create a feature extractor and return (module, feature_dim)."""
    key = name.lower()
    if key == "efficientnet_b4":
        return _create_torchvision_backbone(key, pretrained=pretrained)

    if key == "clip_vit_l14":
        return _create_clip_backbone(key, pretrained=pretrained)

    raise ValueError(
        f"Unsupported backbone '{name}'. Choose one of: {', '.join(sorted(BACKBONE_DIMS.keys()))}."
    )
