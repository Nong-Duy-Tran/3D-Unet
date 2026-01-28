from .simple_unet3d_classifier import SimpleUNet3DClassifier, default_config as simple_default_config
from .unet3d_classifier import UNet3DClassifier, default_config as unet_default_config


def get_model(model_name="simple", **kwargs):
    if model_name == "simple":
        return SimpleUNet3DClassifier(**kwargs)
    if model_name == "unet":
        return UNet3DClassifier(use_residual=False, **kwargs)
    if model_name == "resunet":
        return UNet3DClassifier(use_residual=True, **kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def get_model_config(model_name: str):
    if model_name == "simple":
        return simple_default_config()
    if model_name in {"unet", "resunet"}:
        cfg = unet_default_config()
        cfg["use_residual"] = model_name == "resunet"
        return cfg
    raise ValueError(f"Unknown model: {model_name}")
