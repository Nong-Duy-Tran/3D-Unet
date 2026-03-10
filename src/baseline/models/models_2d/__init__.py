from .deit_classifier import DeiT2DClassifier, default_config as deit2d_default_config
from .simple_unet_classifier import (
    SimpleUNet2DClassifier,
    default_config as simpleunet2d_default_config,
)
from .swin_classifier import Swin2DClassifier, default_config as swin2d_default_config
from .vgg_classifier import VGG2DClassifier, default_config as vgg2d_default_config
from .vit_classifier import ViT2DClassifier, default_config as vit2d_default_config

__all__ = [
    "DeiT2DClassifier",
    "deit2d_default_config",
    "SimpleUNet2DClassifier",
    "simpleunet2d_default_config",
    "Swin2DClassifier",
    "swin2d_default_config",
    "VGG2DClassifier",
    "vgg2d_default_config",
    "ViT2DClassifier",
    "vit2d_default_config",
]
