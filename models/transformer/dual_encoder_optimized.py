import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.transformer.swin_unetr import SwinUNETRSegmentation
from models.transformer.cnn_encoder import CNNEncoder, NFCEBlock

class DualEncoderUNetOptimized(nn.Module):
    """
    Optimized Dual Encoder U-Net that combines SwinUNETR transformer and CNN encoder.
    Features from both encoders are fused using NFCE blocks before being passed to the decoder.
    
    Optimizations:
    - No debug prints in forward pass
    - Option to disable CNN encoder for speed
    - Efficient memory management
    - Gradient checkpointing support
    """
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=3,
        out_channels=4,
        transformer_feature_size=24,  # Reduced default from 48
        cnn_base_filters=8,  # Reduced default from 16
        dropout_rates=[0.1, 0.1, 0.15, 0.15, 0.2],  # Reduced dropout
        use_pretrained=True,
        use_cnn_encoder=True,  # Option to disable CNN encoder
        use_gradient_checkpointing=False
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_cnn_encoder = use_cnn_encoder
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cnn_base_filters = cnn_base_filters if use_cnn_encoder else None
        
        # Initialize transformer encoder (SwinUNETR)
        self.transformer = SwinUNETRSegmentation(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=transformer_feature_size,
            use_checkpoint=use_gradient_checkpointing,
            pretrained=use_pretrained
        )
        
        # Initialize CNN encoder only if needed
        if self.use_cnn_encoder:
            self.cnn_encoder = CNNEncoder(
                in_channels=in_channels,
                base_filters=cnn_base_filters,
                dropout_rates=dropout_rates
            )
            
            # Define feature dimensions
            transformer_channels = [transformer_feature_size, transformer_feature_size, 
                                  transformer_feature_size*2, transformer_feature_size*4]
            # CNN encoder outputs: base_filters, base_filters*2, base_filters*4, base_filters*8
            cnn_channels = [cnn_base_filters, cnn_base_filters*2, 
                           cnn_base_filters*4, cnn_base_filters*8]
            
            # Feature fusion blocks
            self.fusion_blocks = nn.ModuleList([
                NFCEBlock(cnn_channels[i], transformer_channels[i]) 
                for i in range(4)
            ])
        else:
            # When not using CNN encoder, just use transformer features
            transformer_channels = [transformer_feature_size, transformer_feature_size, 
                                  transformer_feature_size*2, transformer_feature_size*4]
        
        # Optimized decoder with fewer parameters
        # Level 3 -> 2
        self.up3 = nn.ConvTranspose3d(transformer_channels[3], transformer_channels[2], 
                                      kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(transformer_channels[2] * 2, 
                                           transformer_channels[2], 
                                           dropout=dropout_rates[2])
        
        # Level 2 -> 1
        self.up2 = nn.ConvTranspose3d(transformer_channels[2], transformer_channels[1], 
                                      kernel_size=2, stride=2)
        self.dec2 = self._make_decoder_block(transformer_channels[1] * 2, 
                                           transformer_channels[1], 
                                           dropout=dropout_rates[1])
        
        # Level 1 -> 0
        self.up1 = nn.ConvTranspose3d(transformer_channels[1], transformer_channels[0], 
                                      kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(transformer_channels[0] * 2, 
                                           transformer_channels[0], 
                                           dropout=dropout_rates[0])
        
        # Final classification layer
        self.final = nn.Conv3d(transformer_channels[0], out_channels, kernel_size=1)
        
    def _make_decoder_block(self, in_channels, out_channels, dropout=0.1):
        """Create a decoder block with reduced complexity."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass through the dual-encoder U-Net.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Segmentation output of shape (B, num_classes, D, H, W)
        """
        # Extract transformer features
        transformer_features = self.transformer.get_features(x)[:4]
        
        if self.use_cnn_encoder:
            # Extract CNN features
            cnn_features = self.cnn_encoder(x)[:4]
            
            # Fuse features
            fused_features = []
            for i, (tf, cf) in enumerate(zip(transformer_features, cnn_features)):
                fused = self.fusion_blocks[i](cf, tf)
                fused_features.append(fused)
            
            # Clear intermediate features
            del transformer_features
            del cnn_features
        else:
            # Use transformer features directly
            fused_features = transformer_features
        
        # Memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decoder path
        x = fused_features[3]
        
        # Level 3 -> 2
        x = self.up3(x)
        if x.shape[2:] != fused_features[2].shape[2:]:
            x = F.interpolate(x, size=fused_features[2].shape[2:], 
                            mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[2]], dim=1)
        x = self.dec3(x)
        
        # Level 2 -> 1
        x = self.up2(x)
        if x.shape[2:] != fused_features[1].shape[2:]:
            x = F.interpolate(x, size=fused_features[1].shape[2:], 
                            mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[1]], dim=1)
        x = self.dec2(x)
        
        # Level 1 -> 0
        x = self.up1(x)
        if x.shape[2:] != fused_features[0].shape[2:]:
            x = F.interpolate(x, size=fused_features[0].shape[2:], 
                            mode='trilinear', align_corners=False)
        x = torch.cat([x, fused_features[0]], dim=1)
        x = self.dec1(x)
        
        # Clear intermediate features
        del fused_features
        
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
                'transformer_feature_size': self.transformer.feature_size,
                'cnn_base_filters': self.cnn_base_filters,
                'use_cnn_encoder': self.use_cnn_encoder,
                'use_gradient_checkpointing': self.use_gradient_checkpointing
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


class SingleEncoderUNet(nn.Module):
    """
    Simplified version using only the SwinUNETR without dual encoding.
    This is significantly faster than the dual encoder version.
    """
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=3,
        out_channels=4,
        feature_size=24,
        use_pretrained=True,
        use_gradient_checkpointing=False
    ):
        super().__init__()
        
        # Just use SwinUNETR directly
        self.model = SwinUNETRSegmentation(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_gradient_checkpointing,
            pretrained=use_pretrained
        )
        
    def forward(self, x):
        return self.model(x)
    
    def save_model(self, path, epoch=None, optimizer=None, loss=None):
        """Save model checkpoint."""
        self.model.save_model(path, epoch, optimizer, loss)
        
    @classmethod
    def load_model(cls, path, device='cuda'):
        """Load model from checkpoint."""
        return SwinUNETRSegmentation.load_model(path, device)
