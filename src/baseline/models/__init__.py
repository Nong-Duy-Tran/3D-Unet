from .factory import get_model, get_model_config
from .models_3d import SimpleUNet3DClassifier, Swin3DClassifier, UNet3DClassifier
from .models_2d import DeiT2DClassifier, SimpleUNet2DClassifier, Swin2DClassifier, VGG2DClassifier, ViT2DClassifier
from .models_25d import Timm25DClassifier, VGG25DClassifier

__all__ = [
    "get_model",
    "get_model_config",
    "SimpleUNet3DClassifier",
    "UNet3DClassifier",
    "ViT2DClassifier",
    "Swin2DClassifier",
    "Swin3DClassifier",
    "DeiT2DClassifier",
    "VGG2DClassifier",
    "SimpleUNet2DClassifier",
    "Timm25DClassifier",
    "VGG25DClassifier",
]
