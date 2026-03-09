# ============================================================================
# EEGNet Model for Motor Imagery EEG Classification
# ============================================================================
# EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces
# Paper: Lawhern et al., 2018
# Adapted for: 8 channels, 250 Hz, 1-second windows (8 x 250)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet model for EEG-based motor imagery classification.
    
    Architecture:
    1. Temporal Convolution: Learns frequency-specific features
    2. Depthwise Convolution: Learns spatial patterns across channels
    3. Separable Convolution: Efficient feature extraction
    4. Classification Head: Final prediction
    
    Input shape: (batch, 1, channels, timepoints) or (batch, channels, timepoints)
    Output shape: (batch, n_classes)
    """
    
    def __init__(
        self,
        n_classes=3,
        n_channels=8,
        n_timepoints=250,
        F1=8,              # Number of temporal filters
        F2=16,             # Number of pointwise filters
        D=2,               # Depth multiplier for depthwise convolution
        kernel_length=64,  # Length of temporal convolution kernel
        pool_time=4,       # Temporal pooling size
        pool_space=2,      # Spatial pooling size (if applicable)
        dropout_rate=0.5,
        sampling_rate=250
    ):
        super(EEGNet, self).__init__()
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        
        # Block 1: Temporal Convolution
        # Input: (batch, 1, channels, timepoints)
        # Output: (batch, F1, channels, timepoints)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False
            ),
            nn.BatchNorm2d(F1)
        )
        
        # Block 2: Depthwise Convolution (spatial filtering)
        # Input: (batch, F1, channels, timepoints)
        # Output: (batch, F1*D, 1, timepoints/pool_time)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,  # Depthwise: each input channel gets its own filter
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_time)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Convolution (pointwise + depthwise)
        # Input: (batch, F1*D, 1, timepoints/pool_time)
        # Output: (batch, F2, 1, timepoints/(pool_time*pool_space))
        self.separable_conv = nn.Sequential(
            # Pointwise convolution
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 1),
                groups=F1 * D,  # Depthwise separable
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            # Depthwise convolution (temporal)
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 16),  # Temporal kernel
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_space)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate flattened size dynamically to avoid rounding errors
        # We'll compute this by passing a dummy tensor through the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal_conv(dummy_input)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            flattened_size = x.numel() // x.size(0)  # Remove batch dimension
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, n_classes)
        )
        
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
        
        # Block 1: Temporal convolution
        x = self.temporal_conv(x)
        
        # Block 2: Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Block 3: Separable convolution
        x = self.separable_conv(x)
        
        # Classification
        output = self.classifier(x)
        
        return output


class EEGNetLite(nn.Module):
    """
    Lightweight version of EEGNet for smaller datasets.
    Uses fewer filters and simpler architecture.
    """
    
    def __init__(
        self,
        n_classes=3,
        n_channels=8,
        n_timepoints=250,
        F1=4,              # Fewer temporal filters
        F2=8,              # Fewer pointwise filters
        D=1,               # Smaller depth multiplier
        kernel_length=32,  # Shorter kernel
        pool_time=4,
        pool_space=2,
        dropout_rate=0.5
    ):
        super(EEGNetLite, self).__init__()
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        
        # Simplified temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False
            ),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )
        
        # Simplified depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_time)),
            nn.Dropout(dropout_rate)
        )
        
        # Simplified separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 8),
                padding=(0, 3),
                bias=False
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_space)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate flattened size dynamically to avoid rounding errors
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal_conv(dummy_input)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            flattened_size = x.numel() // x.size(0)  # Remove batch dimension
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
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
    
    def forward(self, x):
        """Forward pass"""
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        output = self.classifier(x)
        
        return output


# ============================================================================
# Test the model
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("EEGNet Model Test")
    print("=" * 80)
    
    # Model parameters
    n_channels = 8
    n_timepoints = 250
    n_classes = 3
    batch_size = 16
    
    # Create model
    model = EEGNet(
        n_classes=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        F1=8,
        F2=16,
        D=2,
        dropout_rate=0.5
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with 3D input (batch, channels, timepoints)
    dummy_input_3d = torch.randn(batch_size, n_channels, n_timepoints)
    output_3d = model(dummy_input_3d)
    print(f"\n3D Input Test:")
    print(f"  Input shape: {dummy_input_3d.shape}")
    print(f"  Output shape: {output_3d.shape}")
    
    # Test forward pass with 4D input (batch, 1, channels, timepoints)
    dummy_input_4d = torch.randn(batch_size, 1, n_channels, n_timepoints)
    output_4d = model(dummy_input_4d)
    print(f"\n4D Input Test:")
    print(f"  Input shape: {dummy_input_4d.shape}")
    print(f"  Output shape: {output_4d.shape}")
    
    # Test EEGNetLite
    model_lite = EEGNetLite(
        n_classes=n_classes,
        n_channels=n_channels,
        n_timepoints=n_timepoints
    )
    
    lite_params = sum(p.numel() for p in model_lite.parameters())
    print(f"\nEEGNetLite Statistics:")
    print(f"  Total parameters: {lite_params:,}")
    
    output_lite = model_lite(dummy_input_3d)
    print(f"  Output shape: {output_lite.shape}")
    
    print("\n" + "=" * 80)
    print("✅ Models created successfully!")
    print("=" * 80)

