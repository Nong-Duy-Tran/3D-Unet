"""
2D CNN + Simple Attention models for Alzheimer's Disease classification
Uses learned weighted sum instead of complex QKV attention
More suitable for medical imaging with spatial ordering
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    """2D Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock2D, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class CNN2DBackbone(nn.Module):
    """Lightweight 2D CNN backbone for feature extraction"""
    def __init__(self, in_channels=1, base_channels=32):
        super(CNN2DBackbone, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks=2, stride=2)
        # self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_channels * 4
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock2D(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock2D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 1, 224, 224)
        Returns:
            features: Tensor of shape (batch_size, feature_dim)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResBlock2p5D(nn.Module):
    """Residual block using Conv3d with depth kernel size 1 (no cross-slice mixing)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock2p5D, self).__init__()

        spatial_stride = (1, stride, stride)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=spatial_stride,
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=spatial_stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class CNN2p5DBackbone(nn.Module):
    """
    Shared-weight slice backbone using Conv3d with kernel depth=1.

    Input:  (B, C, N, H, W)
    Output: (B, N, feature_dim)
    """
    def __init__(self, in_channels=1, base_channels=32):
        super(CNN2p5DBackbone, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks=2, stride=2)

        self.feature_dim = base_channels * 8

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResBlock2p5D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock2p5D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Keep depth (slice axis) and pool spatial dims only.
        x = x.mean(dim=(-2, -1))      # (B, C, N)
        x = x.transpose(1, 2)          # (B, N, C)
        return x


class SimpleWeightedAttention(nn.Module):
    """
    Simple learned weighted sum attention
    
    Instead of complex QKV attention, this learns a weight (alpha) for each slice
    and computes weighted sum: output = sum(alpha_i * feature_i)
    
    This is more suitable for medical imaging because:
    1. Slices are spatially ordered (don't need to learn relationships)
    2. Some slices are more informative (e.g., hippocampus region for Alzheimer's)
    3. Much more data-efficient (only num_slices parameters)
    4. Interpretable (can visualize which slices the model focuses on)
    
    Args:
        num_slices: Number of slices (120)
        feature_dim: Dimension of features from CNN backbone
    """
    def __init__(self, num_slices, feature_dim):
        super(SimpleWeightedAttention, self).__init__()
        
        self.slice_weights = nn.Parameter(torch.ones(num_slices, 1))
        
        self.weight_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x, use_content_based=True):
        """
        Args:
            x: Input features of shape (batch_size, num_slices, feature_dim)
            use_content_based: If True, compute weights from features; if False, use fixed learned weights
        
        Returns:
            aggregated: Weighted sum of shape (batch_size, feature_dim)
            attention_weights: Weights of shape (batch_size, num_slices) for visualization
        """
        B, N, D = x.shape
        
        if use_content_based:
            slice_scores = self.weight_network(x)  # (B, N, 1)
            attention_weights = F.softmax(slice_scores, dim=1)  # (B, N, 1)
        else:
            attention_weights = F.softmax(self.slice_weights, dim=0)  # (N, 1)
            attention_weights = attention_weights.unsqueeze(0).expand(B, -1, -1)  # (B, N, 1)
        
        # Why sum in dim = 1 ?
        aggregated = (x * attention_weights).sum(dim=1)  # (B, D)
        
        return aggregated, attention_weights.squeeze(-1)


class ThresholdedSimpleWeightedAttention(nn.Module):
    def __init__(self, num_slices, feature_dim, min_attention=1e-3):
        super(ThresholdedSimpleWeightedAttention, self).__init__()

        self.num_slices = num_slices
        self.min_attention = float(min_attention)
        self.slice_weights = nn.Parameter(torch.ones(num_slices, 1))

        self.weight_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x, use_content_based=True):
        B, N, D = x.shape
        if N != self.num_slices:
            raise ValueError(f"Expected {self.num_slices} slices, got {N}")

        if use_content_based:
            slice_scores = self.weight_network(x)
            attention_weights = F.softmax(slice_scores, dim=1)
        else:
            attention_weights = F.softmax(self.slice_weights, dim=0)
            attention_weights = attention_weights.unsqueeze(0).expand(B, -1, -1)

        sparse_weights = torch.where(
            attention_weights < self.min_attention,
            torch.zeros_like(attention_weights),
            attention_weights
        )
        sparse_sum = sparse_weights.sum(dim=1, keepdim=True)
        normalized_sparse = sparse_weights / sparse_sum.clamp_min(1e-12)
        attention_weights = torch.where(sparse_sum > 0, normalized_sparse, attention_weights)

        aggregated = (x * attention_weights).sum(dim=1)
        return aggregated, attention_weights.squeeze(-1)


class GaussianInitializedWeightedAttention(nn.Module):
    def __init__(self, num_slices, sigma_ratio=0.2, trainable=True):
        super(GaussianInitializedWeightedAttention, self).__init__()

        self.num_slices = num_slices
        indices = torch.arange(num_slices, dtype=torch.float32)
        center = (num_slices - 1) / 2.0
        sigma = max(num_slices * float(sigma_ratio), 1.0)
        gaussian = torch.exp(-0.5 * ((indices - center) / sigma) ** 2).unsqueeze(-1)
        init_logits = torch.log((gaussian / gaussian.sum()).clamp_min(1e-8))
        if trainable:
            self.slice_logits = nn.Parameter(init_logits)  # learnable
        else:
            self.register_buffer("slice_logits", init_logits)  # fixed

    def forward(self, x):
        B, N, D = x.shape
        if N != self.num_slices:
            raise ValueError(f"Expected {self.num_slices} slices, got {N}")

        attention_weights = F.softmax(self.slice_logits, dim=0)
        attention_weights = attention_weights.unsqueeze(0).expand(B, -1, -1)
        aggregated = (x * attention_weights).sum(dim=1)
        return aggregated, attention_weights.squeeze(-1)


class SimpleCNN2DAttentionClassifier(nn.Module):
    """
    2D CNN + Simple Weighted Attention classifier
    
    Much simpler than QKV attention:
    - CNN extracts features from each slice
    - Simple weighted sum aggregates across slices
    - Classification head predicts class
    
    Benefits:
    - ~10x fewer parameters in attention
    - More interpretable (can see which slices matter)
    - Better for small medical datasets
    - Faster training and inference
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (2 for binary)
        num_slices: Number of slices per volume (120)
        base_channels: Base channels for CNN backbone (32)
        dropout: Dropout rate (0.3)
        use_content_based: Use content-based attention (True) or fixed weights (False)
        backbone_type: '2d' (flatten B*N slices) or '2p5d' (Conv3d with kernel depth=1)
    """
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=32, dropout=0.3, use_content_based=True, backbone_type='2d'):
        super(SimpleCNN2DAttentionClassifier, self).__init__()
        
        self.num_slices = num_slices
        self.use_content_based = use_content_based
        self.backbone_type = backbone_type

        if backbone_type == '2d':
            self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        elif backbone_type == '2p5d':
            self.backbone = CNN2p5DBackbone(in_channels=in_channels, base_channels=base_channels)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Choose from: 2d, 2p5d")
        
        self.attention = SimpleWeightedAttention(num_slices, self.backbone.feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(self.backbone.feature_dim // 2, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor of shape (batch_size, num_slices, 1, 224, 224)
            return_attention: If True, also return attention weights
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
            attention_weights (optional): Weights of shape (batch_size, num_slices)
        """
        B, N, C, H, W = x.shape

        if self.backbone_type == '2d':
            # Reshape to process all slices in one 2D batch.
            x = x.view(B * N, C, H, W)  # (B*N, 1, 224, 224)
            features = self.backbone(x)  # (B*N, feature_dim)
            features = features.view(B, N, -1)  # (B, N, feature_dim)
        else:
            # Conv3d expects (B, C, D, H, W) where D is the slice axis.
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, 1, N, H, W)
            features = self.backbone(x)  # (B, N, feature_dim)
        
        # Weighted attention aggregation
        aggregated, attention_weights = self.attention(features, self.use_content_based)
        
        # Classification
        logits = self.classifier(aggregated)  # (B, num_classes)
        
        if return_attention:
            return logits, attention_weights
        return logits


class CompactSimpleCNN2DClassifier(nn.Module):
    """
    Compact version with even fewer parameters
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (2 for binary)
        num_slices: Number of slices per volume (120)
        base_channels: Base channels for CNN backbone (24)
        dropout: Dropout rate (0.3)
    """
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=24, dropout=0.3):
        super(CompactSimpleCNN2DClassifier, self).__init__()
        
        self.num_slices = num_slices
        
        # Lightweight CNN backbone
        self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        
        # Simple weighted attention
        self.attention = SimpleWeightedAttention(num_slices, self.backbone.feature_dim)
        
        # Simpler classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        B, N, C, H, W = x.shape
        
        # Extract features
        x = x.view(B * N, C, H, W)
        features = self.backbone(x)
        features = features.view(B, N, -1)
        
        # Attention aggregation
        aggregated, attention_weights = self.attention(features, use_content_based=True)
        
        # Classification
        logits = self.classifier(aggregated)
        
        if return_attention:
            return logits, attention_weights
        return logits


class ThresholdedSimpleCNN2DAttentionClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=32, dropout=0.3, use_content_based=True, min_attention=5e-3):
        super(ThresholdedSimpleCNN2DAttentionClassifier, self).__init__()

        self.num_slices = num_slices
        self.use_content_based = use_content_based

        self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        self.attention = ThresholdedSimpleWeightedAttention(
            num_slices=num_slices,
            feature_dim=self.backbone.feature_dim,
            min_attention=min_attention
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(self.backbone.feature_dim // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        features = self.backbone(x)
        features = features.view(B, N, -1)
        aggregated, attention_weights = self.attention(features, self.use_content_based)
        logits = self.classifier(aggregated)

        if return_attention:
            return logits, attention_weights
        return logits


class GaussianInitSimpleCNN2DAttentionClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=32, dropout=0.3, sigma_ratio=0.2):
        super(GaussianInitSimpleCNN2DAttentionClassifier, self).__init__()

        self.num_slices = num_slices
        self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        self.attention = GaussianInitializedWeightedAttention(
            num_slices=num_slices,
            sigma_ratio=sigma_ratio
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim, self.backbone.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(self.backbone.feature_dim // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        features = self.backbone(x)
        features = features.view(B, N, -1)
        aggregated, attention_weights = self.attention(features)
        logits = self.classifier(aggregated)

        if return_attention:
            return logits, attention_weights
        return logits


class MRIAttentionNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_slices=120, **kwargs):
        super(MRIAttentionNet, self).__init__()

        self.num_slices = num_slices
        feature_dim = 16

        self.shared_cnn = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)  # Reduces each slice to a single feature vector
        )

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 1),  # Looks at the features and gives a score
            nn.Softmax(dim=1)           # Makes all slice scores add up to 1.0
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_attention=False):
        # x shape: (Batch, num_slices, C, H, W) — same as other models
        B, N, C, H, W = x.shape

        # Reshape to process all slices through the same CNN at once
        x = x.view(B * N, C, H, W)  # (B*N, C, H, W)

        # Step 1: Feature Extraction (Shared Weights)
        features = self.shared_cnn(x)        # (B*N, 16, 1, 1)
        features = features.view(B, N, -1)   # (B, N, 16)

        # Step 2: Calculate Attention Scores
        attn_weights = self.attention(features)  # (B, N, 1)

        # Step 3: Weighted Sum
        context_vector = torch.sum(attn_weights * features, dim=1)  # (B, 16)

        # Step 4: Final Prediction
        output = self.classifier(context_vector)  # (B, num_classes)

        if return_attention:
            return output, attn_weights.squeeze(-1)  # (B, N)
        return output


def get_model_2d(model_name='standard', **kwargs):
    """
    Factory function to create 2D classification model with simple attention
    
    Args:
        model_name: 'standard', 'compact', 'mrinet', 'thresholded', or 'gaussian_init'
        **kwargs: Additional arguments for model
    
    Returns:
        model: 2D CNN + Simple Attention classification model
    """
    if model_name == 'standard':
        return SimpleCNN2DAttentionClassifier(**kwargs)
    elif model_name == 'compact':
        return CompactSimpleCNN2DClassifier(**kwargs)
    elif model_name == 'mrinet':
        return MRIAttentionNet(**kwargs)
    elif model_name == 'thresholded':
        return ThresholdedSimpleCNN2DAttentionClassifier(**kwargs)
    elif model_name == 'gaussian_init':
        return GaussianInitSimpleCNN2DAttentionClassifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: standard, compact, mrinet, thresholded, gaussian_init"
        )


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Testing 2D CNN + Simple Attention Models")
    print("=" * 70)
    
    # Test input
    x = torch.randn(2, 80, 1, 224, 224).to(device)
    
    print("\n1. Testing SimpleCNN2DAttentionClassifier...")
    print("-" * 70)
    model_simple = SimpleCNN2DAttentionClassifier(
        in_channels=1,
        num_classes=2,
        num_slices=80,
        base_channels=8,
        dropout=0.3,
        use_content_based=True
    )
    model_simple = model_simple.to(device)
    
    output, attention = model_simple(x, return_attention=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention.shape}")  # [2, 120]
    
    total_params = sum(p.numel() for p in model_simple.parameters())
    trainable_params = sum(p.numel() for p in model_simple.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Show attention weights for first sample
    print(f"\nSample attention weights (first 10 slices): {attention[0, :10].detach().cpu().numpy()}")
    print(f"Sum of attention weights: {attention[0].sum().item():.4f}")  # Should be ~1.0
    
    print("\n2. Testing CompactSimpleCNN2DClassifier...")
    print("-" * 70)
    model_compact = CompactSimpleCNN2DClassifier(
        in_channels=1,
        num_classes=2,
        num_slices=120,
        base_channels=24,
        dropout=0.3
    )
    model_compact = model_compact.to(device)
    
    output = model_compact(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model_compact.parameters())
    trainable_params = sum(p.numel() for p in model_compact.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")