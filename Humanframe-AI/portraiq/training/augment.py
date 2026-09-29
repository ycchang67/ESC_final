
from utils.image_utils import build_model_transform


def build_train_transform(image_size: int, backbone_name: str):
    return build_model_transform(image_size=image_size, backbone_name=backbone_name, train=True)


def build_eval_transform(image_size: int, backbone_name: str):
    return build_model_transform(image_size=image_size, backbone_name=backbone_name, train=False)
