import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.transformer.swin_unetr import SwinUNETRSegmentation
from models.transformer.cnn_encoder import CNNEncoder, NFCEBlock

class DualEncoderUNet(nn.Module):
    """
    Dual Encoder U-Net that combines SwinUNETR transformer and CNN encoder.
    Features from both encoders are fused using NFCE blocks before being passed to the decoder.
    """
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=3,
        out_channels=4,
        transformer_feature_size=48,
        cnn_base_filters=16,
        dropout_rates=[0.1, 0.1, 0.2, 0.2, 0.3],
        use_pretrained=True
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize transformer encoder (SwinUNETR)
        self.transformer = SwinUNETRSegmentation(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,  # We'll use our own decoder, but keep the output classes same
            feature_size=transformer_feature_size,
            pretrained=use_pretrained
        )
        
        # Initialize separate CNN encoder
        self.cnn_encoder = CNNEncoder(
            in_channels=in_channels,
            base_filters=cnn_base_filters,
            dropout_rates=dropout_rates
        )
        
        # Define actual feature dimensions from both encoders
        transformer_channels = [48, 48, 96, 192, 768]  # Actual transformer feature sizes
        cnn_channels = [16, 32, 64, 128, 256]  # Actual CNN feature sizes
        
        # Feature fusion blocks with correct dimensions
        self.fusion_blocks = nn.ModuleList([
            NFCEBlock(cnn_channels[i], transformer_channels[i]) 
            for i in range(5)
        ])
        
        # Add downsampling for level 4 CNN features
        self.cnn_level4_downsample = nn.AvgPool3d(kernel_size=2, stride=2)
        
        # Decoder levels
        # Each decoder level takes input from the fusion block and previous decoder level
        
        # Decoder - First upsampling (from bottom)
        self.up4 = nn.ConvTranspose3d(transformer_channels[4], transformer_channels[3], kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv3d(transformer_channels[3] * 2, transformer_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[3]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(transformer_channels[3], transformer_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - Second upsampling
        self.up3 = nn.ConvTranspose3d(transformer_channels[3], transformer_channels[2], kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(transformer_channels[2] * 2, transformer_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[2]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(transformer_channels[2], transformer_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - Third upsampling
        self.up2 = nn.ConvTranspose3d(transformer_channels[2], transformer_channels[1], kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(transformer_channels[1] * 2, transformer_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[1]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(transformer_channels[1], transformer_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder - Final upsampling
        self.up1 = nn.ConvTranspose3d(transformer_channels[1], transformer_channels[0], kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(transformer_channels[0] * 2, transformer_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[0]),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.1),
            nn.Conv3d(transformer_channels[0], transformer_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(transformer_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.final = nn.Conv3d(transformer_channels[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the dual-encoder U-Net.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Segmentation output of shape (B, num_classes, D, H, W)
        """
        # Extract features from both encoders
        transformer_features = self.transformer.get_features(x)
        cnn_features = self.cnn_encoder(x)
        
        # Debug prints to check feature shapes
        print("Feature shapes at each level:")
        for i, (tf, cf) in enumerate(zip(transformer_features, cnn_features)):
            print(f"Level {i}:")
            print(f"  Transformer: {tf.shape}")
            print(f"  CNN: {cf.shape}")
            
            # Verify spatial dimensions match
            if tf.shape[2:] != cf.shape[2:]:
                print(f"WARNING: Spatial dimension mismatch at level {i}!")
                print(f"Transformer: {tf.shape[2:]}, CNN: {cf.shape[2:]}")
        
        # Fuse features from both encoders
        fused_features = []
        for i, (tf, cf) in enumerate(zip(transformer_features, cnn_features)):
            # Handle spatial dimension mismatch at level 4
            if i == 4:
                cf = self.cnn_level4_downsample(cf)
            
            # Fuse features
            fused = self.fusion_blocks[i](cf, tf)
            fused_features.append(fused)
        
        # Clear intermediate features to save memory
        del transformer_features
        del cnn_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decoder path with shape-adaptive skip connections
        # Level 4 -> 3
        print("\nDecoder shapes:")
        x = self.up4(fused_features[4])
        print(f"After up4: {x.shape}, Target: {fused_features[3].shape}")
        x = F.interpolate(x, size=fused_features[3].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[3]], dim=1)
        x = self.dec4(x)
        
        # Level 3 -> 2
        x = self.up3(x)
        print(f"After up3: {x.shape}, Target: {fused_features[2].shape}")
        x = F.interpolate(x, size=fused_features[2].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[2]], dim=1)
        x = self.dec3(x)
        
        # Level 2 -> 1
        x = self.up2(x)
        print(f"After up2: {x.shape}, Target: {fused_features[1].shape}")
        x = F.interpolate(x, size=fused_features[1].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[1]], dim=1)
        x = self.dec2(x)
        
        # Level 1 -> 0
        x = self.up1(x)
        print(f"After up1: {x.shape}, Target: {fused_features[0].shape}")
        x = F.interpolate(x, size=fused_features[0].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[0]], dim=1)
        x = self.dec1(x)
        
        # Final classification
        x = self.final(x)
        
        return x
    
    def save_model(self, path, epoch=None, optimizer=None, loss=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'img_size': self.img_size,
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                # Add other parameters as needed
            }
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if loss is not None:
            checkpoint['loss'] = loss
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        
    @classmethod
    def load_model(cls, path, device='cuda'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model, checkpoint
