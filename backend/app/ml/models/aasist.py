"""
AASIST Model Architecture for Voice Spoofing Detection

Simplified implementation of Audio Anti-Spoofing using Integrated 
Spectro-Temporal Graph Attention Networks.

Reference: Li et al., "AASIST: Audio Anti-Spoofing using Integrated 
Spectro-Temporal Graph Attention Networks", ICASSP 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block: Conv → BatchNorm → ReLU → Dropout"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, channels, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, (3, 3), padding=1, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, (3, 3), padding=1, dropout=dropout)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out


class SelfAttention(nn.Module):
    """Multi-head self-attention module"""
    
    def __init__(self, channels, num_heads=8):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.head_dim = channels // num_heads
        
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.fc_out = nn.Linear(channels, channels)
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        batch, channels, height, width = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(batch, channels, -1).transpose(1, 2)  # (batch, HW, channels)
        
        # Linear projections
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        
        # Split into multiple heads
        Q = Q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Attention output
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, -1, channels)
        
        # Final linear layer
        out = self.fc_out(out)
        
        # Reshape back to spatial format
        out = out.transpose(1, 2).view(batch, channels, height, width)
        
        return out


class AASIST(nn.Module):
    """
    AASIST Model for Voice Spoofing Detection
    
    Architecture:
    1. Convolutional frontend (multi-scale feature extraction)
    2. Residual blocks (deep feature learning)
    3. Self-attention (spectro-temporal modeling)
    4. Global average pooling
    5. Classification head
    """
    
    def __init__(self, num_classes=2, dropout=0.2, num_heads=8):
        super(AASIST, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional Frontend
        # Input: (batch, 1, 128, time_steps)
        self.conv1 = ConvBlock(1, 32, (3, 3), padding=1, dropout=dropout)  # (batch, 32, 128, time_steps)
        self.pool1 = nn.MaxPool2d((2, 2))  # (batch, 32, 64, time_steps/2)
        
        self.conv2 = ConvBlock(32, 64, (3, 3), padding=1, dropout=dropout)  # (batch, 64, 64, time_steps/2)
        self.pool2 = nn.MaxPool2d((2, 2))  # (batch, 64, 32, time_steps/4)
        
        self.conv3 = ConvBlock(64, 128, (3, 3), padding=1, dropout=dropout)  # (batch, 128, 32, time_steps/4)
        self.pool3 = nn.MaxPool2d((2, 2))  # (batch, 128, 16, time_steps/8)
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(128, dropout=dropout)
        self.res_block2 = ResidualBlock(128, dropout=dropout)
        
        # Self-Attention Layer
        self.attention = SelfAttention(128, num_heads=num_heads)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.fc1 = nn.Linear(128, 256)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 1, 128, time_steps)
               where 128 is number of mel-frequency bins
               
        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # Convolutional frontend
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Self-attention
        x = self.attention(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        
        return logits


class LossFunction(nn.Module):
    """Combined loss function for voice spoofing detection"""
    
    def __init__(self, loss_type='cross_entropy'):
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AASIST Model Test")
    print("="*70)
    
    # Create model
    model = AASIST(num_classes=2)
    print(f"\n✓ Model created successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 501)  # (batch, channels, mel_bins, time_steps)
    
    print(f"\n📍 Testing forward pass...")
    print(f"  Input shape: {x.shape}")
    
    logits = model(x)
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, 2)")
    
    # Test loss calculation
    labels = torch.LongTensor([0, 1, 0, 1])
    loss_fn = LossFunction(loss_type='cross_entropy')
    loss = loss_fn(logits, labels)
    
    print(f"\n✓ Loss calculation successful!")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test backward pass
    loss.backward()
    print(f"\n✓ Backward pass successful!")
    
    print(f"\n✅ AASIST model test PASSED!")
