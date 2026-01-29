from .simple_unet3d_classifier import SimpleUNet3DClassifier, default_config as simple_default_config
from .unet3d_classifier import UNet3DClassifier, default_config as unet_default_config
from .vit2d_classifier import ViT2DClassifier, default_config as vit2d_default_config
from .swin2d_classifier import Swin2DClassifier, default_config as swin2d_default_config
from .swin3d_classifier import Swin3DClassifier, default_config as swin3d_default_config
from .deit2d_classifier import DeiT2DClassifier, default_config as deit2d_default_config


def get_model(model_name="simple", **kwargs):
    if model_name == "simple":
        return SimpleUNet3DClassifier(**kwargs)
    if model_name == "unet":
        return UNet3DClassifier(use_residual=False, **kwargs)
    if model_name == "resunet":
        return UNet3DClassifier(use_residual=True, **kwargs)
    if model_name == "vit2d":
        return ViT2DClassifier(**kwargs)
    if model_name == "swin2d":
        return Swin2DClassifier(**kwargs)
    if model_name == "swin3d":
        return Swin3DClassifier(**kwargs)
    if model_name == "deit2d":
        return DeiT2DClassifier(**kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def get_model_config(model_name: str):
    if model_name == "simple":
        return simple_default_config()
    if model_name in {"unet", "resunet"}:
        cfg = unet_default_config()
        cfg["use_residual"] = model_name == "resunet"
        return cfg
    if model_name == "vit2d":
        return vit2d_default_config()
    if model_name == "swin2d":
        return swin2d_default_config()
    if model_name == "swin3d":
        return swin3d_default_config()
    if model_name == "deit2d":
        return deit2d_default_config()
    raise ValueError(f"Unknown model: {model_name}")
