"""
Classification model using pytorch-3dunet encoder
Adapts the segmentation U-Net for classification tasks
"""
import torch
import torch.nn as nn
from pytorch3dunet.unet3d.model import UNet3D, ResidualUNet3D
from monai.networks.nets import SwinUNETR

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


class CompactUNet3DClassifier(nn.Module):
    """
    Compact 3D U-Net for classification with 2-3M parameters
    Lighter version of SimpleUNet3DClassifier for faster training
    
    Args:
        in_channels: Number of input channels (default: 1 for MRI)
        num_classes: Number of output classes (default: 2 for binary)
        base_features: Base number of features (default: 24)
    """
    
    def __init__(self, in_channels=1, num_classes=2, base_features=24):
        super(CompactUNet3DClassifier, self).__init__()
        
        # Encoder with 4 levels instead of 5
        self.enc1 = self._make_encoder(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._make_encoder(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._make_encoder(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._make_encoder(base_features * 4, base_features * 8)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier with fewer parameters
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(base_features * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def _make_encoder(self, in_channels, out_channels):
        """Create compact encoder block with single conv + batch norm"""
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
        
        # Bottleneck
        x4 = self.bottleneck(self.pool3(x3))
        
        # Global pooling and classification
        pooled = self.global_pool(x4)
        logits = self.classifier(pooled)
        
        return logits


class SwinUNet3DClassifier(nn.Module):
    """
    Swin-UNETR adapted for classification
    Uses transformer-based Swin-UNETR encoder + global pooling + classifier head
    
    Args:
        in_channels: Number of input channels (default: 1 for MRI)
        num_classes: Number of output classes (default: 2 for binary)
        feature_size: Base feature size (default: 48)
        depths: Depths of each Swin Transformer layer (default: (2, 2, 2, 2))
        num_heads: Number of attention heads in each layer (default: (3, 6, 12, 24))
        window_size: Window size for Swin Transformer (default: 7)
        drop_rate: Dropout rate (default: 0.0)
        attn_drop_rate: Attention dropout rate (default: 0.0)
    """
    
    def __init__(self, img_size=(96, 96, 96), in_channels=1, num_classes=2, 
                 feature_size=48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 drop_rate=0.0, attn_drop_rate=0.0, window_size=7, **kwargs):
        super(SwinUNet3DClassifier, self).__init__()
        
        # Swin-UNETR backbone for feature extraction
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,  # Dummy, we won't use the decoder output
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=False,  # Set to True if memory is a concern
        )
        
        # Store parameters for later use
        self.feature_size = feature_size
        self.depths = depths
        self.img_size = img_size
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Determine actual bottleneck features by running a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, *img_size)
            hidden_states_out = self.swin_unetr.swinViT(dummy_input, self.swin_unetr.normalize)
            bottleneck_features = hidden_states_out[-1]
            pooled = self.global_pool(bottleneck_features)
            actual_features = pooled.view(pooled.size(0), -1).shape[1]
        
        # Classification head - no nn.Flatten() since we manually flatten in forward()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(actual_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass using Swin Transformer encoder for classification
        
        Args:
            x: Input tensor (batch_size, channels, depth, height, width)
        
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Extract features using Swin-UNETR encoder
        # Access the encoder path to get deep features
        hidden_states_out = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)
        
        # Get the bottleneck features (last encoder output)
        # hidden_states_out is a list of feature maps from different stages
        bottleneck_features = hidden_states_out[-1]
        
        # Global pooling and flatten
        pooled = self.global_pool(bottleneck_features)
        pooled_flat = torch.flatten(pooled, 1)
        
        # Classify
        logits = self.classifier(pooled_flat)
        
        return logits


def get_model(model_name='simple', **kwargs):
    """
    Factory function to create classification model
    
    Args:
        model_name: 'simple', 'compact', 'unet', 'resunet', or 'swinunet'
        **kwargs: Additional arguments for model
    
    Returns:
        model: Classification model
    """
    if model_name == 'simple':
        return SimpleUNet3DClassifier(**kwargs)
    elif model_name == 'compact':
        return CompactUNet3DClassifier(**kwargs)
    elif model_name == 'unet':
        return UNet3DClassifier(use_residual=False, **kwargs)
    elif model_name == 'resunet':
        return UNet3DClassifier(use_residual=True, **kwargs)
    elif model_name == 'swinunet':
        return SwinUNet3DClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: simple, compact, unet, resunet, swinunet")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Compact U-Net Classifier...")
    model_compact = CompactUNet3DClassifier(in_channels=1, num_classes=2, base_features=24)
    model_compact = model_compact.to(device)
    
    # Test input
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    output = model_compact(x)
    print(f"Output shape: {output.shape}")  # Should be [2, 2]
    
    # Count parameters
    total_params = sum(p.numel() for p in model_compact.parameters())
    print(f"Total parameters (Compact): {total_params:,}")
    
    print("\nTesting Simple U-Net Classifier...")
    model_simple = SimpleUNet3DClassifier(in_channels=1, num_classes=2, base_features=32)
    model_simple = model_simple.to(device)
    output = model_simple(x)
    print(f"Output shape: {output.shape}")  # Should be [2, 2]
    
    total_params = sum(p.numel() for p in model_simple.parameters())
    print(f"Total parameters (Simple): {total_params:,}")
    
    print("\nTesting UNet3D Classifier...")
    model_unet = UNet3DClassifier(in_channels=1, num_classes=2, f_maps=32, num_levels=4)
    model_unet = model_unet.to(device)
    output = model_unet(x)
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model_unet.parameters())
    print(f"Total parameters (UNet3D): {total_params:,}")

    print("\nTesting SwinUNet3D Classifier...")
    model_swin = SwinUNet3DClassifier(
        img_size=(64, 64, 64),  # Match test input size
        in_channels=1, 
        num_classes=2, 
        feature_size=24,  # Smaller for testing
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=4  # Smaller window for 64x64x64 input
    )
    model_swin = model_swin.to(device)
    output = model_swin(x)
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model_swin.parameters())
    print(f"Total parameters (SwinUNet3D): {total_params:,}")
