"""
2D CNN + Attention models for Alzheimer's Disease classification
Processes 120 axial slices (224x224) with attention-based aggregation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock2D(nn.Module):
    """
    2D Residual Block
    """
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
    """
    Lightweight 2D CNN backbone for feature extraction from single slices
    Similar to a small ResNet architecture
    
    Args:
        in_channels: Number of input channels (1 for grayscale MRI)
        base_channels: Base number of feature channels (default: 32)
    
    Returns:
        Features of shape (batch_size, feature_dim)
    """
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


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            out: Output tensor of shape (batch_size, seq_len, embed_dim)
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class CNN2DAttentionClassifier(nn.Module):
    """
    2D CNN + Attention classifier for Alzheimer's Disease
    
    Architecture:
    1. 2D CNN backbone processes each slice independently
    2. Multi-head attention aggregates features across slices
    3. Classification head predicts class
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (2 for binary)
        num_slices: Number of slices per volume (120)
        base_channels: Base channels for CNN backbone (32)
        num_heads: Number of attention heads (8)
        embed_dim: Embedding dimension (256)
        dropout: Dropout rate (0.3)
    """
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=32, num_heads=8, embed_dim=256, dropout=0.3):
        super(CNN2DAttentionClassifier, self).__init__()
        
        self.num_slices = num_slices
        
        # CNN backbone for slice feature extraction
        self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        
        # Project CNN features to embedding dimension
        self.feature_proj = nn.Linear(self.backbone.feature_dim, embed_dim)
        
        # Learnable positional embeddings for slices
        self.pos_embedding = nn.Parameter(torch.randn(1, num_slices, embed_dim))
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Global pooling across slices
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_slices, 1, 224, 224)
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
        """
        B, N, C, H, W = x.shape
        
        # Reshape to process all slices in batch
        x = x.view(B * N, C, H, W)  # (B*N, 1, 224, 224)
        
        # Extract features from each slice
        features = self.backbone(x)  # (B*N, feature_dim)
        
        # Reshape back to (batch, slices, features)
        features = features.view(B, N, -1)  # (B, N, feature_dim)
        
        # Project to embedding dimension
        features = self.feature_proj(features)  # (B, N, embed_dim)
        
        # Add positional embeddings
        features = features + self.pos_embedding
        
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(self.norm1(features))
        features = features + attn_out
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(self.norm2(features))
        features = features + ffn_out  # (B, N, embed_dim)
        
        # Global pooling across slices
        features = features.permute(0, 2, 1)  # (B, embed_dim, N)
        pooled = self.global_pool(features).squeeze(-1)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits


class CompactCNN2DAttentionClassifier(nn.Module):
    """
    Compact version of CNN2DAttentionClassifier with fewer parameters
    Suitable for faster training and smaller datasets
    
    Args:
        in_channels: Number of input channels (1 for grayscale)
        num_classes: Number of output classes (2 for binary)
        num_slices: Number of slices per volume (120)
        base_channels: Base channels for CNN backbone (24)
        num_heads: Number of attention heads (4)
        embed_dim: Embedding dimension (128)
        dropout: Dropout rate (0.3)
    """
    def __init__(self, in_channels=1, num_classes=2, num_slices=120,
                 base_channels=24, num_heads=4, embed_dim=128, dropout=0.3):
        super(CompactCNN2DAttentionClassifier, self).__init__()
        
        self.num_slices = num_slices
        
        # Lightweight CNN backbone
        self.backbone = CNN2DBackbone(in_channels=in_channels, base_channels=base_channels)
        
        # Project CNN features to embedding dimension
        self.feature_proj = nn.Linear(self.backbone.feature_dim, embed_dim)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_slices, embed_dim))
        
        # Single attention layer
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simpler classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_slices, 1, 224, 224)
        
        Returns:
            logits: Class logits of shape (batch_size, num_classes)
        """
        B, N, C, H, W = x.shape
        
        # Reshape and extract features
        x = x.view(B * N, C, H, W)
        features = self.backbone(x)
        features = features.view(B, N, -1)
        
        # Project and add positional embeddings
        features = self.feature_proj(features)
        features = features + self.pos_embedding
        
        # Attention
        attn_out, _ = self.attention(self.norm(features))
        features = features + attn_out
        
        # Global pooling and classification
        features = features.permute(0, 2, 1)
        pooled = self.global_pool(features).squeeze(-1)
        logits = self.classifier(pooled)
        
        return logits


def get_model_2d(model_name='standard', **kwargs):
    """
    Factory function to create 2D classification model
    
    Args:
        model_name: 'standard' or 'compact'
        **kwargs: Additional arguments for model
    
    Returns:
        model: 2D CNN + Attention classification model
    """
    if model_name == 'standard':
        return CNN2DAttentionClassifier(**kwargs)
    elif model_name == 'compact':
        return CompactCNN2DAttentionClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: standard, compact")


if __name__ == "__main__":
    # Test models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Testing 2D CNN + Attention Models")
    print("=" * 70)
    
    # Test input: (batch_size=2, num_slices=120, channels=1, height=224, width=224)
    x = torch.randn(2, 120, 1, 224, 224).to(device)
    
    print("\n1. Testing Standard CNN2DAttentionClassifier...")
    print("-" * 70)
    model_standard = CNN2DAttentionClassifier(
        in_channels=1,
        num_classes=2,
        num_slices=120,
        base_channels=32,
        num_heads=8,
        embed_dim=256,
        dropout=0.3
    )
    model_standard = model_standard.to(device)
    
    output = model_standard(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [2, 2]
    
    total_params = sum(p.numel() for p in model_standard.parameters())
    trainable_params = sum(p.numel() for p in model_standard.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n2. Testing Compact CNN2DAttentionClassifier...")
    print("-" * 70)
    model_compact = CompactCNN2DAttentionClassifier(
        in_channels=1,
        num_classes=2,
        num_slices=120,
        base_channels=24,
        num_heads=4,
        embed_dim=128,
        dropout=0.3
    )
    model_compact = model_compact.to(device)
    
    output = model_compact(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [2, 2]
    
    total_params = sum(p.numel() for p in model_compact.parameters())
    trainable_params = sum(p.numel() for p in model_compact.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n3. Testing model factory...")
    print("-" * 70)
    model_factory = get_model_2d('compact', num_slices=120)
    model_factory = model_factory.to(device)
    output = model_factory(x)
    print(f"Factory model output shape: {output.shape}")
    
    print("\n" + "=" * 70)
    print("Model tests completed successfully!")
    print("=" * 70)
    
    # Print model architecture summary
    print("\nModel architecture (compact):")
    print(model_compact)
