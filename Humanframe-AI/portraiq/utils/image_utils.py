
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

try:
    from ..models.backbone import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
except ImportError:  # pragma: no cover - script execution fallback
    from models.backbone import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_rgb_array(image_pil: Image.Image):
    return np.array(image_pil)


def _norm_for_backbone(backbone_name: str): # characterization of the backbone
    name = str(backbone_name).lower()
    if name.startswith("clip_"):
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    return IMAGENET_MEAN, IMAGENET_STD


def build_model_transform(image_size: int, backbone_name: str, train: bool = False):
    mean, std = _norm_for_backbone(backbone_name)
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)
