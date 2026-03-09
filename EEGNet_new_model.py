# ============================================================================
# Improved EEGNet Model for Motor Imagery EEG Classification
# ============================================================================
# Enhanced version with:
# - More filters (F1=16 vs 8)
# - Longer temporal kernel (0.5s = 125 samples)
# - Spatial dropout support
# - Better regularization
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout2d(nn.Module):
    """
    Spatial dropout - drops entire channels (2D feature maps) instead of individual elements.
    Better regularization for EEG data.
    """
    def __init__(self, p=0.5):
        super(SpatialDropout2d, self).__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # x shape: (batch, channels, height, width)
        # Drop entire channels (feature maps)
        batch_size, num_channels, height, width = x.size()
        
        # Create mask: (batch, channels, 1, 1)
        mask = torch.ones(batch_size, num_channels, 1, 1, device=x.device)
        mask = F.dropout2d(mask, p=self.p, training=True)
        
        return x * mask


class EEGNet(nn.Module):
    """
    Improved EEGNet model for EEG-based motor imagery classification.
    
    Key improvements:
    - More temporal filters (F1=16)
    - Longer temporal kernel (0.5s = 125 samples at 250 Hz)
    - Optional spatial dropout
    - Better regularization
    
    Input shape: (batch, 1, channels, timepoints) or (batch, channels, timepoints)
    Output shape: (batch, n_classes)
    """
    
    def __init__(
        self,
        num_classes=3,
        num_channels=8,
        num_samples=250,  # Can be 250 (1s) or 750 (3s) or 1000 (4s)
        fs=250,            # Sampling rate
        dropout_rate=0.5,
        F1=16,             # Number of temporal filters (increased from 8)
        D=2,               # Depth multiplier
        F2=None,           # Number of pointwise filters (auto = F1 if None)
        k_temporal=None,   # Temporal kernel length (auto = 0.5s if None)
        use_spatial_dropout=True,  # Use spatial dropout instead of regular dropout
    ):
        super(EEGNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.fs = fs
        
        # Auto-calculate temporal kernel (0.5 seconds)
        if k_temporal is None:
            k_temporal = int(fs * 0.5)  # 0.5 seconds = 125 samples at 250 Hz
        self.k_temporal = k_temporal
        
        # Auto-calculate F2
        if F2 is None:
            F2 = F1  # Default: same as F1
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1: Temporal Convolution
        # Input: (batch, 1, channels, timepoints)
        # Output: (batch, F1, channels, timepoints)
        self.firstconv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, k_temporal),
                padding=(0, k_temporal // 2),
                bias=False
            ),
            nn.BatchNorm2d(F1)
        )
        
        # Block 2: Depthwise Convolution (spatial filtering)
        # Input: (batch, F1, channels, timepoints)
        # Output: (batch, F1*D, 1, timepoints/4)
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(num_channels, 1),
                groups=F1,  # Depthwise: each input channel gets its own filter
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # Reduce time by 4x
        )
        
        # Apply dropout (spatial or regular)
        if use_spatial_dropout:
            self.dropout1 = SpatialDropout2d(p=dropout_rate)
        else:
            self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable Convolution
        # Input: (batch, F1*D, 1, timepoints/4)
        # Output: (batch, F2, 1, timepoints/32)
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 16),  # Temporal kernel
                padding=(0, 8),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),  # Reduce time by 8x
        )
        
        # Apply dropout
        if use_spatial_dropout:
            self.dropout2 = SpatialDropout2d(p=dropout_rate)
        else:
            self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, num_samples)
            x = self.firstconv(dummy_input)
            x = self.depthwiseConv(x)
            x = self.dropout1(x)
            x = self.separableConv(x)
            x = self.dropout2(x)
            flattened_size = x.numel() // x.size(0)  # Remove batch dimension
        
        self.flattened_size = flattened_size
        
        # Classification head
        self.classify = nn.Linear(flattened_size, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _forward_features(self, x):
        """
        Forward pass through convolutional layers only.
        Used for feature extraction and size calculation.
        """
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.dropout1(x)
        x = self.separableConv(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, timepoints) or (batch, 1, channels, timepoints)
        
        Returns:
            output: Class predictions (batch, n_classes)
        """
        # Handle input shape: if 3D, add channel dimension
        if len(x.shape) == 3:
            # (batch, channels, timepoints) -> (batch, 1, channels, timepoints)
            x = x.unsqueeze(1)
        
        # Forward through feature extraction
        x = self._forward_features(x)
        
        # Classification
        output = self.classify(x)
        
        return output


# ============================================================================
# Test the model
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Improved EEGNet Model Test")
    print("=" * 80)
    
    # Test with different input sizes
    batch_size = 16
    n_channels = 8
    n_classes = 3
    
    # Test 1: 1-second window (250 samples)
    print("\nTest 1: 1-second window (250 samples)")
    model_1s = EEGNet(
        num_classes=n_classes,
        num_channels=n_channels,
        num_samples=250,
        fs=250,
        F1=16,
        D=2,
        dropout_rate=0.5,
        use_spatial_dropout=True
    )
    
    total_params = sum(p.numel() for p in model_1s.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    dummy_input = torch.randn(batch_size, 1, n_channels, 250)
    output = model_1s(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test 2: 3-second window (750 samples)
    print("\nTest 2: 3-second window (750 samples)")
    model_3s = EEGNet(
        num_classes=n_classes,
        num_channels=n_channels,
        num_samples=750,
        fs=250,
        F1=16,
        D=2,
        dropout_rate=0.5,
        use_spatial_dropout=True
    )
    
    dummy_input_3s = torch.randn(batch_size, 1, n_channels, 750)
    output_3s = model_3s(dummy_input_3s)
    print(f"  Input shape: {dummy_input_3s.shape}")
    print(f"  Output shape: {output_3s.shape}")
    
    # Test 3: 3D input (auto-adds channel dimension)
    print("\nTest 3: 3D input (auto-adds channel dimension)")
    dummy_input_3d = torch.randn(batch_size, n_channels, 250)
    output_3d = model_1s(dummy_input_3d)
    print(f"  Input shape: {dummy_input_3d.shape}")
    print(f"  Output shape: {output_3d.shape}")
    
    print("\n" + "=" * 80)
    print("✅ Model tests passed!")
    print("=" * 80)

