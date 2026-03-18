import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    densenet121, DenseNet121_Weights, 
    resnet50, ResNet50_Weights, 
    resnet18, ResNet18_Weights
)
from torchinfo import summary as Summary


class MyDenseNetMultiAttention(nn.Module):
    """
    DenseNet-based slice encoder with self-attention aggregation for
    3D MRI classification.

    Pipeline:
        1. Slice-wise feature extraction using pretrained DenseNet121
        2. Slice embedding projection
        3. Multi-head self-attention across slices
        4. Global slice pooling
        5. Classification head

    Expected input shape:
        (B, S, 1, H, W)

        B = batch size
        S = number of slices per volume
    """

    def __init__(self, num_classes=2, num_slices=80, embed_dim=256):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. CNN Backbone (DenseNet121 pretrained on ImageNet)
        # ------------------------------------------------------------------
        base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Convert first convolution layer to accept grayscale input (1 channel)
        original_conv = base_model.features.conv0
        base_model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Initialize grayscale weights by summing RGB pretrained weights
        base_model.features.conv0.weight.data = original_conv.weight.data.sum(
            dim=1, keepdim=True
        )

        self.backbone = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------------------------------
        # 2. Slice-level Attention Module
        # ------------------------------------------------------------------
        self.proj = nn.Linear(1024, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # 3. Classification Head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: MRI volume tensor
               Shape = (B, S, 1, H, W)

        Returns:
            logits: classification output (B, num_classes)
            attn_weights: attention map across slices
                          Shape = (B, num_heads, S, S)
        """

        B, S, C, H, W = x.shape

        # --------------------------------------------------------------
        # Slice-wise CNN feature extraction
        # --------------------------------------------------------------
        x = x.view(B * S, C, H, W)

        features = self.backbone(x)          # (B*S, 1024, H', W')
        features = self.avgpool(features)    # (B*S, 1024, 1, 1)
        features = features.flatten(1)       # (B*S, 1024)

        # Restore slice sequence structure
        features = features.view(B, S, 1024)

        # --------------------------------------------------------------
        # Slice embedding projection
        # --------------------------------------------------------------
        features = self.proj(features)       # (B, S, embed_dim)

        # --------------------------------------------------------------
        # Multi-head self-attention across slices
        # --------------------------------------------------------------
        attn_out, attn_weights = self.attn(
            features,
            features,
            features,
        )

        # Residual connection + global slice pooling
        out = (features + attn_out).mean(dim=1)   # (B, embed_dim)

        # --------------------------------------------------------------
        # Classification
        # --------------------------------------------------------------
        logits = self.classifier(out)

        return logits, attn_weights 
    
class MyResNetMultiAttention(nn.Module):
    """
    ResNet-based slice encoder with self-attention aggregation for
    3D MRI classification.

    Pipeline:
        1. Slice-wise feature extraction using pretrained ResNet50
        2. Slice embedding projection
        3. Multi-head self-attention across slices
        4. Global slice pooling
        5. Classification head

    Expected input shape:
        (B, S, 1, H, W)

        B = batch size
        S = number of slices per volume
    """

    def __init__(self, num_classes=2, num_slices=80, embed_dim=256):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. CNN Backbone (ResNet50 pretrained on ImageNet)
        # ------------------------------------------------------------------
        # base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        base_model = resnet18(weights=None)

        self.feature_dim = base_model.fc.in_features
        

        # Convert first convolution layer to accept grayscale input (1 channel)
        original_conv = base_model.conv1
        base_model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Initialize grayscale weights by summing RGB pretrained weights
        base_model.conv1.weight.data = original_conv.weight.data.sum(
            dim=1, keepdim=True
        )

        # Remove the final FC layer; keep everything up to the global avg pool
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------------------------------------------------
        # 2. Slice-level Attention Module
        # ------------------------------------------------------------------
        self.proj = nn.Linear(self.feature_dim, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # 3. Classification Head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: MRI volume tensor
               Shape = (B, S, 1, H, W)

        Returns:
            logits: classification output (B, num_classes)
            attn_weights: attention map across slices
                          Shape = (B, num_heads, S, S)
        """

        B, S, C, H, W = x.shape

        # --------------------------------------------------------------
        # Slice-wise CNN feature extraction
        # --------------------------------------------------------------
        x = x.view(B * S, C, H, W)

        features = self.backbone(x)          # (B*S, 2048, H', W')
        features = self.avgpool(features)    # (B*S, 2048, 1, 1)
        features = features.flatten(1)       # (B*S, 2048)

        # Restore slice sequence structure
        features = features.view(B, S, self.feature_dim)

        # --------------------------------------------------------------
        # Slice embedding projection
        # --------------------------------------------------------------
        features = self.proj(features)       # (B, S, embed_dim)

        # --------------------------------------------------------------
        # Multi-head self-attention across slices
        # --------------------------------------------------------------
        attn_out, attn_weights = self.attn(
            features,
            features,
            features,
        )

        # Residual connection + global slice pooling
        out = (features + attn_out).mean(dim=1)   # (B, embed_dim)

        # --------------------------------------------------------------
        # Classification
        # --------------------------------------------------------------
        logits = self.classifier(out)

        return logits, attn_weights
    

class MyMobileNetMultiAttention(nn.Module):
    def __init__(self, num_classes=2, num_slices=80, embed_dim=256):
        super().__init__()

        # Lightweight backbone with internal skip connections (MobileNetV3 Small)
        base_model = mobilenet_v3_small(weights=None)

        # Modify the first convolutional layer to accept single-channel (grayscale) MRI input
        base_model.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)

        self.backbone = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV3 outputs  feature channels
        self.feature_dim = base_model.classifier[0].in_features
        self.proj = nn.Linear(self.feature_dim, embed_dim)

        # Positional embeddings for slices (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Multi-head self-attention module for slice-level feature aggregation
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head with normalization and dropout for regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.5),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        B, S, C, H, W = x.shape
        # Process all slices in parallel by reshaping to (B*S, C, H, W)
        x = x.view(B * S, C, H, W)

        features = self.backbone(x)
        features = self.avgpool(features).flatten(1)

        # Reshape features to (B, S, feature_dim) for sequence modeling
        features = features.view(B, S, -1)
        embeddings = self.proj(features)

        embeddings = embeddings + self.pos_embed



        # Apply self-attention across slices
        attn_out, attn_weights = self.attn(embeddings, embeddings, embeddings)

        # Combine embeddings and attention output, then pool across slices
        out = (embeddings + attn_out).mean(dim=1)
        logits = self.classifier(out)

        return logits, attn_weights


class MyEfficientNetMultiAttention(nn.Module):
    """
    EfficientNet-based slice encoder with self-attention aggregation for
    3D MRI classification.
    
    EfficientNet provides a better accuracy-to-efficiency trade-off compared to
    other models. Uses EfficientNet-B0 as backbone.
    
    Pipeline:
        1. Slice-wise feature extraction using pretrained EfficientNet-B0
        2. Slice embedding projection
        3. Learnable positional embeddings
        4. Multi-head self-attention across slices
        5. Global slice pooling
        6. Classification head
    
    Expected input shape:
        (B, S, 1, H, W)
        
        B = batch size
        S = number of slices per volume
    """
    
    def __init__(self, num_classes=2, num_slices=80, embed_dim=256):
        super().__init__()
        
        # ------------------------------------------------------------------
        # 1. CNN Backbone (EfficientNet-B0 pretrained on ImageNet)
        # ------------------------------------------------------------------
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Convert first convolution layer to accept grayscale input (1 channel)
        original_conv = base_model.features[0][0]
        base_model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        
        # Initialize grayscale weights by summing RGB pretrained weights
        base_model.features[0][0].weight.data = original_conv.weight.data.sum(
            dim=1, keepdim=True
        )
        
        self.backbone = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # EfficientNet-B0 outputs 1280 channels
        self.feature_dim = 1280
        
        # ------------------------------------------------------------------
        # 2. Slice-level Attention Module
        # ------------------------------------------------------------------
        self.proj = nn.Linear(self.feature_dim, embed_dim)
        
        # Positional embeddings for slices (learnable)
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_slices, embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
        )
        
        # ------------------------------------------------------------------
        # 3. Classification Head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.4),
            nn.Linear(embed_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for projection and classification layers."""
        for module in [self.proj, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: MRI volume tensor
               Shape = (B, S, 1, H, W)
        
        Returns:
            logits: classification output (B, num_classes)
            attn_weights: attention map across slices
                          Shape = (B, num_heads, S, S)
        """
        B, S, C, H, W = x.shape
        
        # --------------------------------------------------------------
        # Slice-wise CNN feature extraction
        # --------------------------------------------------------------
        x = x.view(B * S, C, H, W)
        
        features = self.backbone(x)          # (B*S, 1280, H', W')
        features = self.avgpool(features)    # (B*S, 1280, 1, 1)
        features = features.flatten(1)       # (B*S, 1280)
        
        # Restore slice sequence structure
        features = features.view(B, S, self.feature_dim)
        
        # --------------------------------------------------------------
        # Slice embedding projection
        # --------------------------------------------------------------
        features = self.proj(features)       # (B, S, embed_dim)
        
        # Add positional embeddings
        features = features + self.pos_embed
        
        # --------------------------------------------------------------
        # Multi-head self-attention across slices
        # --------------------------------------------------------------
        attn_out, attn_weights = self.attn(
            features,
            features,
            features,
        )
        
        # Residual connection + global slice pooling
        out = (features + attn_out).mean(dim=1)   # (B, embed_dim)
        
        # --------------------------------------------------------------
        # Classification
        # --------------------------------------------------------------
        logits = self.classifier(out)
        
        return logits, attn_weights


if __name__ == "__main__":
    print("=" * 70)
    print("=" * 70)
    model = MyMobileNetMultiAttention(num_classes=2, num_slices=80, embed_dim=256)
    print(model)
    Summary(model, input_size=(1, 80, 1, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"])
    
