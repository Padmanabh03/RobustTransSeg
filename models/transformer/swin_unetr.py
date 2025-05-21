import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRSegmentation(nn.Module):
    """
    A wrapper for MONAI's SwinUNETR model for 3D medical image segmentation.
    Includes pretrained weights for better initialization.
    """
    def __init__(
        self,
        img_size=(128, 128, 128),
        in_channels=3,
        out_channels=4,
        feature_size=48,
        use_checkpoint=True,
        pretrained=True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        
        # Initialize SwinUNETR model from MONAI
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint
        )
        
        # Load pretrained weights if available
        if pretrained:
            try:
                weights = torch.hub.load_state_dict_from_url(
                    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt",
                    map_location="cpu"
                )
                self.model.load_from(weights=weights)
                print("Pretrained SwinUNETR weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights. Error: {e}")
                print("Initializing with random weights instead.")
                
    def forward(self, x):
        """
        Forward pass through the SwinUNETR model.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
                
        Returns:
            Segmentation output of shape (B, num_classes, D, H, W)
        """
        return self.model(x)
    
    def get_features(self, x, levels=None):
        """
        Extract intermediate features for feature fusion.
        
        Args:
            x: Input tensor
            levels: Which feature levels to return (default is all levels)
            
        Returns:
            List of features at different resolutions
        """
        # Run encoder part and collect features
        features = []
        
        # Debug print to check input shape
        print(f"Input shape: {x.shape}")
        
        try:
            # Level 0: Initial convolution (full resolution)
            x0 = self.model.encoder1(x)
            features.append(x0)
            print(f"Feature level 0 shape: {x0.shape}")
            
            # Get transformer features using MONAI's built-in encoder
            # Start with patch embedding
            hidden = self.model.swinViT.patch_embed(x)
            hidden = self.model.swinViT.pos_drop(hidden)
            
            # Level 1: First encoder level (1/2 resolution)
            x1 = self.model.encoder2(hidden)
            features.append(x1)
            print(f"Feature level 1 shape: {x1.shape}")
            
            # Level 2: Process through first transformer layer blocks (1/4 resolution)
            layer1_hidden = hidden
            for blk in self.model.swinViT.layers1:
                layer1_hidden = blk(layer1_hidden)
            x2 = self.model.encoder3(layer1_hidden)
            features.append(x2)
            print(f"Feature level 2 shape: {x2.shape}")
            
            # Level 3: Process through second transformer layer blocks (1/8 resolution)
            layer2_hidden = layer1_hidden
            for blk in self.model.swinViT.layers2:
                layer2_hidden = blk(layer2_hidden)
            x3 = self.model.encoder4(layer2_hidden)
            features.append(x3)
            print(f"Feature level 3 shape: {x3.shape}")
            
            # Level 4: Process through remaining transformer layer blocks (1/16 resolution)
            layer3_hidden = layer2_hidden
            for blk in self.model.swinViT.layers3:
                layer3_hidden = blk(layer3_hidden)
            
            layer4_hidden = layer3_hidden
            for blk in self.model.swinViT.layers4:
                layer4_hidden = blk(layer4_hidden)
            
            x4 = self.model.encoder10(layer4_hidden)
            features.append(x4)
            print(f"Feature level 4 shape: {x4.shape}")
            
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            raise
        
        # Return requested feature levels
        if levels is not None:
            return [features[i] for i in levels]
        
        return features
    
    def save_model(self, path, epoch=None, optimizer=None, loss=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'img_size': self.img_size,
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'feature_size': self.feature_size
            }
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if loss is not None:
            checkpoint['loss'] = loss
            
        torch.save(checkpoint, path)
        
    @classmethod
    def load_model(cls, path, device='cuda'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model, checkpoint
