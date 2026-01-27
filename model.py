"""
Classification model using pytorch-3dunet encoder
Adapts the segmentation U-Net for classification tasks
"""
import torch
import torch.nn as nn
from pytorch3dunet.unet3d.model import UNet3D, ResidualUNet3D


class UNet3DClassifier(nn.Module):
    """
    3D U-Net adapted for classification
    Uses U-Net encoder + global pooling + classifier head
    
    Args:
        in_channels: Number of input channels (default: 1 for MRI)
        num_classes: Number of output classes (default: 2 for binary)
        f_maps: Base number of feature maps (default: 64)
        num_levels: Number of encoder/decoder levels (default: 4)
        use_residual: Use residual blocks instead of double conv (default: False)
    """
    
    def __init__(self, in_channels=1, num_classes=2, f_maps=64, num_levels=4, use_residual=False):
        super(UNet3DClassifier, self).__init__()
        
        # Use pytorch-3dunet's U-Net as feature extractor
        # We'll use the encoder part only
        if use_residual:
            self.unet = ResidualUNet3D(
                in_channels=in_channels,
                out_channels=num_classes,  # Dummy, we won't use decoder output
                final_sigmoid=False,
                f_maps=f_maps,
                num_levels=num_levels,
                is_segmentation=True
            )
        else:
            self.unet = UNet3D(
                in_channels=in_channels,
                out_channels=num_classes,  # Dummy, we won't use decoder output
                final_sigmoid=False,
                f_maps=f_maps,
                num_levels=num_levels,
                is_segmentation=True
            )
        
        # Calculate the number of features from the bottleneck
        # For U-Net with default settings, bottleneck features = f_maps * (2^(num_levels-1))
        bottleneck_features = f_maps * (2 ** (num_levels - 1))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.num_levels = num_levels
    
    def forward(self, x):
        """
        Forward pass - use encoder features for classification
        
        Args:
            x: Input tensor (batch_size, channels, depth, height, width)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Get encoder features
        # We'll extract features from the deepest encoder layer (bottleneck)
        encoders_features = []
        
        # Pass through encoders
        for encoder in self.unet.encoders:
            x = encoder(x)
            encoders_features.append(x)
        
        # Use the last encoder output (bottleneck features)
        bottleneck_features = encoders_features[-1]
        
        # Global pooling
        pooled = self.global_pool(bottleneck_features)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class SimpleUNet3DClassifier(nn.Module):
    """
    Simplified 3D U-Net for classification
    Custom implementation without using full U-Net decoder
    """
    
    def __init__(self, in_channels=1, num_classes=2, base_features=32):
        super(SimpleUNet3DClassifier, self).__init__()
        
        # Encoder
        self.enc1 = self._make_encoder(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._make_encoder(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._make_encoder(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = self._make_encoder(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._make_encoder(base_features * 8, base_features * 16)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(base_features * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_encoder(self, in_channels, out_channels):
        """Create encoder block"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        
        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))
        
        # Global pooling and classification
        pooled = self.global_pool(x5)
        logits = self.classifier(pooled)
        
        return logits


def get_model(model_name='simple', **kwargs):
    """
    Factory function to create classification model
    
    Args:
        model_name: 'simple', 'unet', or 'resunet'
        **kwargs: Additional arguments for model
    
    Returns:
        model: Classification model
    """
    if model_name == 'simple':
        return SimpleUNet3DClassifier(**kwargs)
    elif model_name == 'unet':
        return UNet3DClassifier(use_residual=False, **kwargs)
    elif model_name == 'resunet':
        return UNet3DClassifier(use_residual=True, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Simple U-Net Classifier...")
    model_simple = SimpleUNet3DClassifier(in_channels=1, num_classes=2, base_features=32)
    model_simple = model_simple.to(device)
    
    # Test input
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    output = model_simple(x)
    print(f"Output shape: {output.shape}")  # Should be [2, 2]
    
    # Count parameters
    total_params = sum(p.numel() for p in model_simple.parameters())
    print(f"Total parameters (Simple): {total_params:,}")
    
    print("\nTesting UNet3D Classifier...")
    model_unet = UNet3DClassifier(in_channels=1, num_classes=2, f_maps=32, num_levels=4)
    model_unet = model_unet.to(device)
    output = model_unet(x)
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model_unet.parameters())
    print(f"Total parameters (UNet3D): {total_params:,}")
