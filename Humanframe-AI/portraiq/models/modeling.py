
from typing import Dict, Tuple

import torch.nn as nn

try:
    from .backbone import create_backbone
    from .scorer import CompositionModel, RegressionHead
except ImportError:  # pragma: no cover - script execution fallback
    from models.backbone import create_backbone
    from models.scorer import CompositionModel, RegressionHead


def build_composition_model(config: Dict) -> Tuple[nn.Module, Dict[str, int]]:
    backbone_name = config["model"]["backbone"]
    pretrained = config["model"].get("pretrained", True)

    backbone, feature_dim = create_backbone(backbone_name, pretrained=pretrained)
    scorer = RegressionHead(input_dim=feature_dim)
    model = CompositionModel(backbone=backbone, scorer_head=scorer)

    return model, {"feature_dim": feature_dim}
