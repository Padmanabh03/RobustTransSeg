import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basic convolutional block for the CNN encoder.
    Includes two 3D convolutions with batch normalization.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        
        # Initialize weights using He initialization
        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_p)
        self.relu = nn.ReLU(inplace=True)
        
        # Apply He initialization
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class CNNEncoder(nn.Module):
    """
    Standalone CNN encoder designed for feature extraction at multiple scales.
    This encoder can work in parallel with the transformer encoder.
    """
    def __init__(self, in_channels=3, base_filters=16, dropout_rates=[0.1, 0.1, 0.2, 0.2, 0.3]):
        super().__init__()
        
        # Encoder blocks at different resolutions
        self.enc1 = ConvBlock(in_channels, base_filters, dropout_p=dropout_rates[0])
        self.enc2 = ConvBlock(base_filters, base_filters*2, dropout_p=dropout_rates[1])
        self.enc3 = ConvBlock(base_filters*2, base_filters*4, dropout_p=dropout_rates[2])
        self.enc4 = ConvBlock(base_filters*4, base_filters*8, dropout_p=dropout_rates[3])
        self.enc5 = ConvBlock(base_filters*8, base_filters*16, dropout_p=dropout_rates[4])
        
        # Pooling for downsampling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        """
        Forward pass through the CNN encoder.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            List of feature maps at different resolutions
        """
        # Store features at each level
        features = []
        
        # Encoder path with feature extraction
        enc1 = self.enc1(x)
        features.append(enc1)
        
        enc2 = self.enc2(self.pool(enc1))
        features.append(enc2)
        
        enc3 = self.enc3(self.pool(enc2))
        features.append(enc3)
        
        enc4 = self.enc4(self.pool(enc3))
        features.append(enc4)
        
        enc5 = self.enc5(self.pool(enc4))
        features.append(enc5)
        
        return features

class NFCEBlock(nn.Module):
    """
    Novel Feature Cross-Enhancement Block for fusing CNN and transformer features.
    Based on the architecture shown in the UNETR diagram.
    """
    def __init__(self, cnn_channels, transformer_channels):
        super().__init__()
        
        # Ensure channels match for fusion
        self.cnn_proj = None
        if cnn_channels != transformer_channels:
            self.cnn_proj = nn.Conv3d(cnn_channels, transformer_channels, kernel_size=1)
        
        # Depth-wise separable convolution
        self.dw_conv = nn.Sequential(
            nn.Conv3d(transformer_channels, transformer_channels, kernel_size=3, padding=1, groups=transformer_channels),
            nn.BatchNorm3d(transformer_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(transformer_channels, transformer_channels, kernel_size=1),
            nn.BatchNorm3d(transformer_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1 projection
        self.final_proj = nn.Conv3d(transformer_channels, transformer_channels, kernel_size=1)
        
    def forward(self, cnn_feat, transformer_feat):
        """
        Fuse CNN and transformer features.
        
        Args:
            cnn_feat: Feature from CNN encoder
            transformer_feat: Feature from transformer encoder
            
        Returns:
            Fused feature map
        """
        # Check shape compatibility
        if cnn_feat.shape[2:] != transformer_feat.shape[2:]:
            print(f"Shape mismatch in NFCE block: CNN {cnn_feat.shape}, Transformer {transformer_feat.shape}")
            # Resize the CNN feature to match transformer feature spatial dimensions
            cnn_feat = F.interpolate(
                cnn_feat, 
                size=transformer_feat.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
            print(f"Resized CNN feature to: {cnn_feat.shape}")
        
        # Project CNN features if needed
        if self.cnn_proj is not None:
            cnn_feat = self.cnn_proj(cnn_feat)
        
        try:
            # Add features
            combined = cnn_feat + transformer_feat
            
            # Apply depth-wise convolution
            enhanced = self.dw_conv(combined)
            
            # Final projection with residual connection
            output = combined + self.final_proj(enhanced)
            
            return output
        except RuntimeError as e:
            print(f"Error in NFCE fusion: {str(e)}")
            print(f"CNN feature shape: {cnn_feat.shape}")
            print(f"Transformer feature shape: {transformer_feat.shape}")
            raise
