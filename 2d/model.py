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
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks=2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_channels * 8
    
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
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
        
        # Learnable weight for each slice
        self.slice_weights = nn.Parameter(torch.ones(num_slices, 1))
        
        # Optional: Add a small network to compute weights from features
        # This allows content-based weighting (not just position-based)
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
            # Compute attention weights from features (content-based)
            # Each slice gets a weight based on its content
            slice_scores = self.weight_network(x)  # (B, N, 1)
            attention_weights = F.softmax(slice_scores, dim=1)  # (B, N, 1)
        else:
            # Use fixed learned weights (position-based)
            # Same weights for all samples in batch
            attention_weights = F.softmax(self.slice_weights, dim=0)  # (N, 1)
            attention_weights = attention_weights.unsqueeze(0).expand(B, -1, -1)  # (B, N, 1)
        
        # Weighted sum: multiply each slice by its weight and sum
        aggregated = (x * attention_weights).sum(dim=1)  # (B, D)
        
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
    """
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=32, dropout=0.3, use_content_based=True):
        super(SimpleCNN2DAttentionClassifier, self).__init__()
        
        self.num_slices = num_slices
        self.use_content_based = use_content_based
        
        # CNN backbone for slice feature extraction
        self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        
        # Simple weighted attention
        self.attention = SimpleWeightedAttention(num_slices, self.backbone.feature_dim)
        
        # Classification head
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
        
        # Reshape to process all slices in batch
        x = x.view(B * N, C, H, W)  # (B*N, 1, 224, 224)
        
        # Extract features from each slice
        features = self.backbone(x)  # (B*N, feature_dim)
        
        # Reshape back to (batch, slices, features)
        features = features.view(B, N, -1)  # (B, N, feature_dim)
        
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


class MRIAttentionNet(nn.Module):
    def __init__(self):
        super(MRIAttentionNet, self).__init__()
        
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1) # Reduces each slice to a single feature vector
        )
        
        self.attention = nn.Sequential(
            nn.Linear(16, 1), # Looks at the features and gives a score
            nn.Softmax(dim=1) # Makes all 96 scores add up to 100% (1.0)
        )
        
        self.classifier = nn.Linear(16, 2) # e.g., Healthy vs. Disease

    def forward(self, x):
        # x shape: [Batch, 96, 96, 96] -> (Batch, Slices, H, W)
        batch_size, num_slices, h, w = x.shape
        
        # Reshape to process all slices through the same CNN at once
        # New shape: [Batch * 96, 1, 96, 96]
        x = x.view(batch_size * num_slices, 1, h, w)
        
        # Step 1: Feature Extraction (Shared Weights)
        features = self.shared_cnn(x) # Shape: [Batch*96, 16, 1, 1]
        features = features.view(batch_size, num_slices, 16)
        
        # Step 2: Calculate Attention Scores (Distribution)
        # This tells you how important each of the 96 slides is
        attn_weights = self.attention(features) # Shape: [Batch, 96, 1]
        
        # Step 3: Weighted Sum (Combine the 96 slices into one)
        context_vector = torch.sum(attn_weights * features, dim=1) # [Batch, 16]
        
        # Step 4: Final Prediction
        output = self.classifier(context_vector)
        
        return output, attn_weights


def get_model_2d(model_name='standard', **kwargs):
    """
    Factory function to create 2D classification model with simple attention
    
    Args:
        model_name: 'standard' or 'compact'
        **kwargs: Additional arguments for model
    
    Returns:
        model: 2D CNN + Simple Attention classification model
    """
    if model_name == 'standard':
        return SimpleCNN2DAttentionClassifier(**kwargs)
    elif model_name == 'compact':
        return CompactSimpleCNN2DClassifier(**kwargs)
    elif model_name == "mrinet":
        return MRIAttentionNet(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: standard, compact")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Testing 2D CNN + Simple Attention Models")
    print("=" * 70)
    
    # Test input
    x = torch.randn(2, 120, 1, 224, 224).to(device)
    
    print("\n1. Testing SimpleCNN2DAttentionClassifier...")
    print("-" * 70)
    model_simple = SimpleCNN2DAttentionClassifier(
        in_channels=1,
        num_classes=2,
        num_slices=120,
        base_channels=32,
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