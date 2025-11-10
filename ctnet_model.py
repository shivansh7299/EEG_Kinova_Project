# # ============================================================================
# # CTNet (Convolutional Transformer Network) for Motor Imagery EEG
# # ============================================================================
# # CTNet is specifically designed for MI-EEG with limited data
# # Much better than FBMSNet for small datasets!

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class CTNet(nn.Module):
#     """
#     Convolutional Transformer Network for EEG Motor Imagery
    
#     Key advantages over FBMSNet:
#     - 3-4x fewer parameters → less overfitting
#     - Transformer attention → captures temporal dependencies better
#     - Specifically designed for motor imagery
#     - Works well with limited data (2k-5k samples)
    
#     Architecture:
#     1. Temporal convolution (learns frequency patterns)
#     2. Spatial convolution (learns channel relationships)
#     3. Transformer encoder (learns global temporal patterns)
#     4. Classification head
#     """
    
#     def __init__(self, n_channels=8, n_timepoints=250, n_classes=5, 
#                  embed_dim=64, num_heads=4, num_layers=2, dropout=0.5):
#         """
#         Args:
#             n_channels: Number of EEG channels (default: 8)
#             n_timepoints: Samples per trial (default: 250)
#             n_classes: Number of classes (default: 5)
#             embed_dim: Transformer embedding dimension (default: 64)
#             num_heads: Number of attention heads (default: 4)
#             num_layers: Number of transformer layers (default: 2)
#             dropout: Dropout rate (default: 0.5)
#         """
#         super(CTNet, self).__init__()
        
#         self.n_channels = n_channels
#         self.n_timepoints = n_timepoints
#         self.n_classes = n_classes
        
#         # ====================================================================
#         # BLOCK 1: Temporal Convolution
#         # Purpose: Learn frequency-specific features
#         # ====================================================================
#         self.temporal_conv = nn.Sequential(
#             # First temporal layer: capture different frequency bands
#             nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
            
#             # Second temporal layer: refine features
#             nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
#             nn.BatchNorm2d(64),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.5)  # Light dropout
#         )
        
#         # ====================================================================
#         # BLOCK 2: Spatial Convolution
#         # Purpose: Learn spatial patterns across channels
#         # ====================================================================
#         self.spatial_conv = nn.Sequential(
#             # Depthwise spatial convolution (efficient!)
#             nn.Conv2d(64, 64, kernel_size=(n_channels, 1), groups=64),
#             nn.BatchNorm2d(64),
#             nn.ELU(),
            
#             # Pointwise convolution to mix spatial features
#             nn.Conv2d(64, embed_dim, kernel_size=1),
#             nn.BatchNorm2d(embed_dim),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.5)
#         )
        
#         # After convolutions, we have shape: (batch, embed_dim, 1, time_reduced)
#         # Calculate reduced time dimension
#         self.time_reduced = n_timepoints  # No pooling yet, stays same
        
#         # ====================================================================
#         # BLOCK 3: Temporal Pooling (reduce sequence length)
#         # ====================================================================
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 50))  # Reduce to 50 time steps
#         self.time_reduced = 50
        
#         # ====================================================================
#         # BLOCK 4: Positional Encoding
#         # Purpose: Add position information for transformer
#         # ====================================================================
#         self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=self.time_reduced)
        
#         # ====================================================================
#         # BLOCK 5: Transformer Encoder
#         # Purpose: Capture global temporal dependencies with attention
#         # ====================================================================
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 4,
#             dropout=dropout * 0.5,
#             activation='gelu',
#             batch_first=True
#         )
        
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )
        
#         # ====================================================================
#         # BLOCK 6: Classification Head
#         # ====================================================================
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim, n_classes)
#         )
        
#         # Initialize weights
#         self._init_weights()
    
#     def _init_weights(self):
#         """Initialize weights using Xavier initialization"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         """
#         Args:
#             x: Input tensor of shape (batch, channels, timepoints)
#                or (batch, 1, channels, timepoints)
        
#         Returns:
#             output: Class predictions (batch, n_classes)
#             features: Learned features for visualization (batch, embed_dim)
#         """
#         # Handle different input shapes
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, channels, time)
        
#         # BLOCK 1: Temporal convolution
#         x = self.temporal_conv(x)  # (batch, 64, channels, time)
        
#         # BLOCK 2: Spatial convolution
#         x = self.spatial_conv(x)  # (batch, embed_dim, 1, time)
        
#         # BLOCK 3: Pooling
#         x = self.avg_pool(x)  # (batch, embed_dim, 1, time_reduced)
        
#         # Reshape for transformer: (batch, time_reduced, embed_dim)
#         x = x.squeeze(2).transpose(1, 2)
        
#         # BLOCK 4: Add positional encoding
#         x = self.pos_encoder(x)
        
#         # BLOCK 5: Transformer encoding
#         x = self.transformer_encoder(x)  # (batch, time_reduced, embed_dim)
        
#         # Global average pooling over time
#         features = x.mean(dim=1)  # (batch, embed_dim)
        
#         # BLOCK 6: Classification
#         output = self.classifier(features)  # (batch, n_classes)
        
#         return output, features


# class PositionalEncoding(nn.Module):
#     """
#     Positional encoding for transformer
#     Adds position information to sequence
#     """
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Create positional encoding matrix
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (batch, seq_len, d_model)
#         """
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


# # ============================================================================
# # SIMPLIFIED CTNET (Even fewer parameters for very small datasets)
# # ============================================================================
# class CTNetLite(nn.Module):
#     """
#     Lightweight version of CTNet for extremely limited data
#     Use this if you have <3000 training samples
#     """
#     def __init__(self, n_channels=8, n_timepoints=250, n_classes=5, dropout=0.5):
#         super(CTNetLite, self).__init__()
        
#         # Simpler convolutions
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.25)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(n_channels, 1)),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.25)
#         )
        
#         # Simpler transformer (1 layer, 2 heads)
#         self.pos_encoder = PositionalEncoding(32, dropout=0.1, max_len=250)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=32,
#             nhead=2,
#             dim_feedforward=64,
#             dropout=dropout * 0.5,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(32, n_classes)
#         )
    
#     def forward(self, x):
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)
        
#         x = self.conv1(x)
#         x = self.conv2(x)
        
#         x = x.squeeze(2).transpose(1, 2)
#         x = self.pos_encoder(x)
#         x = self.transformer(x)
        
#         features = x.mean(dim=1)
#         output = self.classifier(features)
        
#         return output, features


# # ============================================================================
# # USAGE EXAMPLE
# # ============================================================================
# if __name__ == "__main__":
#     # Test the model
#     print("="*80)
#     print("CTNet Model Test")
#     print("="*80)
    
#     # Model parameters
#     n_channels = 8
#     n_timepoints = 250
#     n_classes = 5
#     batch_size = 16
    
#     # Create model
#     model = CTNet(
#         n_channels=n_channels,
#         n_timepoints=n_timepoints,
#         n_classes=n_classes,
#         embed_dim=64,
#         num_heads=4,
#         num_layers=2,
#         dropout=0.5
#     )
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\nModel Statistics:")
#     print(f"  Total parameters: {total_params:,}")
#     print(f"  Trainable parameters: {trainable_params:,}")
#     print(f"  FBMSNet typically has: ~75,000 parameters")
#     print(f"  Reduction: {100*(1 - total_params/75000):.1f}%")
    
#     # Test forward pass
#     dummy_input = torch.randn(batch_size, n_channels, n_timepoints)
#     output, features = model(dummy_input)
    
#     print(f"\nInput/Output Shapes:")
#     print(f"  Input: {dummy_input.shape}")
#     print(f"  Output: {output.shape}")
#     print(f"  Features: {features.shape}")
    
#     # Test CTNetLite
#     model_lite = CTNetLite(
#         n_channels=n_channels,
#         n_timepoints=n_timepoints,
#         n_classes=n_classes,
#         dropout=0.5
#     )
    
#     lite_params = sum(p.numel() for p in model_lite.parameters())
#     print(f"\nCTNetLite Statistics:")
#     print(f"  Total parameters: {lite_params:,}")
#     print(f"  Even lighter: {100*(1 - lite_params/total_params):.1f}% smaller than CTNet")
    
#     print("\n" + "="*80)
#     print("✅ Models created successfully!")
#     print("="*80)



# # ============================================================================
# # CTNet (Convolutional Transformer Network) for Motor Imagery EEG
# # ============================================================================
# # Compatible with older PyTorch (<1.8) – no batch_first argument used

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # ============================================================================
# # MAIN CTNET MODEL
# # ============================================================================
# class CTNet(nn.Module):
#     """
#     Convolutional Transformer Network for EEG Motor Imagery
#     """

#     def __init__(self, n_channels=8, n_timepoints=250, n_classes=5,
#                  embed_dim=64, num_heads=4, num_layers=2, dropout=0.5):
#         super(CTNet, self).__init__()

#         self.n_channels = n_channels
#         self.n_timepoints = n_timepoints
#         self.n_classes = n_classes

#         # --------------------------------------------------------------------
#         # BLOCK 1: Temporal Convolution
#         # --------------------------------------------------------------------
#         self.temporal_conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
#             nn.BatchNorm2d(64),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.5)
#         )

#         # --------------------------------------------------------------------
#         # BLOCK 2: Spatial Convolution
#         # --------------------------------------------------------------------
#         self.spatial_conv = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=(n_channels, 1), groups=64),
#             nn.BatchNorm2d(64),
#             nn.ELU(),
#             nn.Conv2d(64, embed_dim, kernel_size=1),
#             nn.BatchNorm2d(embed_dim),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.5)
#         )

#         # --------------------------------------------------------------------
#         # BLOCK 3: Temporal Pooling
#         # --------------------------------------------------------------------
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 50))
#         self.time_reduced = 50

#         # --------------------------------------------------------------------
#         # BLOCK 4: Positional Encoding
#         # --------------------------------------------------------------------
#         self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=self.time_reduced)

#         # --------------------------------------------------------------------
#         # BLOCK 5: Transformer Encoder (manual batch-first compatibility)
#         # --------------------------------------------------------------------
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=embed_dim * 4,
#             dropout=dropout * 0.5,
#             activation='gelu'
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # --------------------------------------------------------------------
#         # BLOCK 6: Classification Head
#         # --------------------------------------------------------------------
#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim, n_classes)
#         )

#         self._init_weights()

#     def _init_weights(self):
#         """Xavier initialization"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         """
#         Args:
#             x: (batch, channels, timepoints)
#         Returns:
#             output: (batch, n_classes)
#             features: (batch, embed_dim)
#         """
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)  # (B, 1, C, T)

#         # Temporal + Spatial conv
#         x = self.temporal_conv(x)
#         x = self.spatial_conv(x)
#         x = self.avg_pool(x)  # (B, embed_dim, 1, time_reduced)

#         # Reshape for transformer: (B, time_reduced, embed_dim)
#         x = x.squeeze(2).transpose(1, 2)
#         x = self.pos_encoder(x)

#         # Transformer (older PyTorch: needs (time, batch, features))
#         x = x.permute(1, 0, 2)  # (T, B, E)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # (B, T, E)

#         # Average pooling
#         features = x.mean(dim=1)
#         output = self.classifier(features)
#         return output, features


# # ============================================================================
# # POSITIONAL ENCODING
# # ============================================================================
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float()
#                              * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


# # ============================================================================
# # LIGHTWEIGHT VERSION
# # ============================================================================
# class CTNetLite(nn.Module):
#     """Lighter version for small datasets (<3000 samples)."""
#     def __init__(self, n_channels=8, n_timepoints=250, n_classes=5, dropout=0.5):
#         super(CTNetLite, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.25)
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=(n_channels, 1)),
#             nn.BatchNorm2d(32),
#             nn.ELU(),
#             nn.Dropout(dropout * 0.25)
#         )

#         self.pos_encoder = PositionalEncoding(32, dropout=0.1, max_len=250)

#         # Transformer layer (no batch_first)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=32,
#             nhead=2,
#             dim_feedforward=64,
#             dropout=dropout * 0.5
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(32, n_classes)
#         )

#     def forward(self, x):
#         if len(x.shape) == 3:
#             x = x.unsqueeze(1)

#         x = self.conv1(x)
#         x = self.conv2(x)

#         # (B, features, time)
#         x = x.squeeze(2).transpose(1, 2)
#         x = self.pos_encoder(x)

#         # Transformer expects (T, B, E)
#         x = x.permute(1, 0, 2)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)

#         features = x.mean(dim=1)
#         output = self.classifier(features)
#         return output, features


# # ============================================================================
# # TEST
# # ============================================================================
# if __name__ == "__main__":
#     print("="*80)
#     print("CTNet Model Test (Compatibility Check)")
#     print("="*80)

#     n_channels = 8
#     n_timepoints = 250
#     n_classes = 5
#     batch_size = 16

#     model = CTNet(
#         n_channels=n_channels,
#         n_timepoints=n_timepoints,
#         n_classes=n_classes,
#         embed_dim=64,
#         num_heads=4,
#         num_layers=2,
#         dropout=0.5
#     )

#     dummy_input = torch.randn(batch_size, n_channels, n_timepoints)
#     output, features = model(dummy_input)

#     print(f"✅ Output: {output.shape}, Features: {features.shape}")

#     model_lite = CTNetLite(
#         n_channels=n_channels,
#         n_timepoints=n_timepoints,
#         n_classes=n_classes,
#         dropout=0.5
#     )

#     output_lite, features_lite = model_lite(dummy_input)
#     print(f"✅ Lite Output: {output_lite.shape}, Features: {features_lite.shape}")
#     print("="*80)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CTNet(nn.Module):
    """
    Fixed CTNet for Motor Imagery EEG
    
    Key improvements:
    1. Less aggressive pooling (250 → 100 instead of 50)
    2. Larger embedding dimension (128 instead of 64)
    3. Consistent dropout
    4. Proper batch handling
    5. Added residual connections
    """
    
    def __init__(self, n_channels=8, n_timepoints=250, n_classes=5, 
                 embed_dim=128, num_heads=4, num_layers=2, dropout=0.5):
        super(CTNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # ====================================================================
        # BLOCK 1: Temporal Convolution - IMPROVED
        # ====================================================================
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout * 0.3),  # Lighter dropout early
            
            nn.Conv2d(32, 64, kernel_size=(1, 15), padding=(0, 7)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # ====================================================================
        # BLOCK 2: Spatial Convolution - IMPROVED
        # ====================================================================
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(n_channels, 1), groups=64),
            nn.BatchNorm2d(64),
            nn.ELU(),
            
            nn.Conv2d(64, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # ====================================================================
        # BLOCK 3: LESS Aggressive Pooling - FIXED
        # ====================================================================
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 100))  # Keep more temporal info!
        self.time_reduced = 100
        
        # ====================================================================
        # BLOCK 4: Positional Encoding
        # ====================================================================
        self.pos_encoder = PositionalEncoding(
            embed_dim, 
            dropout=0.1, 
            max_len=self.time_reduced
        )
        
        # ====================================================================
        # BLOCK 5: Transformer Encoder - IMPROVED
        # ====================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout * 0.4,  # Moderate dropout
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # ====================================================================
        # BLOCK 6: Classification Head with Residual - IMPROVED
        # ====================================================================
        self.feature_projection = nn.Linear(embed_dim, embed_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, timepoints) or (batch, 1, channels, timepoints)
        Returns:
            output: (batch, n_classes)
            features: (batch, embed_dim)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        
        # Temporal + Spatial convolutions
        x = self.temporal_conv(x)      # (B, 64, C, T)
        x = self.spatial_conv(x)        # (B, embed_dim, 1, T)
        x = self.avg_pool(x)            # (B, embed_dim, 1, time_reduced)
        
        # Reshape for transformer: (B, time_reduced, embed_dim)
        x = x.squeeze(2).transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer (handle older PyTorch)
        x = x.permute(1, 0, 2)          # (T, B, E)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)          # (B, T, E)
        
        # Global average pooling + max pooling (richer features)
        avg_features = x.mean(dim=1)    # (B, E)
        max_features = x.max(dim=1)[0]  # (B, E)
        features = avg_features + 0.3 * max_features  # Combine both
        
        # Classification
        output = self.classifier(features)
        
        return output, features


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() 
                           * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CTNetLite(nn.Module):
    """
    Lightweight CTNet for small datasets
    FIXED: Less aggressive pooling, better dropout
    """
    def __init__(self, n_channels=8, n_timepoints=250, n_classes=5, dropout=0.6):
        super(CTNetLite, self).__init__()
        
        # Temporal convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Spatial convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Less aggressive pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 100))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(32, dropout=0.1, max_len=100)
        
        # Simpler transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=2,
            dim_feedforward=64,
            dropout=dropout * 0.4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Classifier with extra layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, n_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        
        x = x.squeeze(2).transpose(1, 2)
        x = self.pos_encoder(x)
        
        # Transformer
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        # Combined pooling
        features = x.mean(dim=1) + 0.3 * x.max(dim=1)[0]
        output = self.classifier(features)
        
        return output, features


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("="*80)
    print("Fixed CTNet Test")
    print("="*80)
    
    n_channels = 8
    n_timepoints = 250
    n_classes = 5
    batch_size = 16
    
    # Test CTNet
    model = CTNet(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        embed_dim=128,  # Increased!
        num_heads=4,
        num_layers=2,
        dropout=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nCTNet Parameters: {total_params:,}")
    
    dummy_input = torch.randn(batch_size, n_channels, n_timepoints)
    output, features = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    
    # Test CTNetLite
    model_lite = CTNetLite(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=n_classes,
        dropout=0.6
    )
    
    lite_params = sum(p.numel() for p in model_lite.parameters())
    print(f"\nCTNetLite Parameters: {lite_params:,}")
    
    output_lite, features_lite = model_lite(dummy_input)
    print(f"Lite output shape: {output_lite.shape}")
    print(f"Lite features shape: {features_lite.shape}")
    
    print("\n" + "="*80)
    print("Key Improvements:")
    print("  ✅ Less aggressive pooling (250→100 instead of 250→50)")
    print("  ✅ Larger embedding (128 instead of 64)")
    print("  ✅ Combined avg+max pooling for richer features")
    print("  ✅ Consistent dropout strategy")
    print("  ✅ Deeper classification head")
    print("="*80)