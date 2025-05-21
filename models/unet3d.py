import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(ConvBlock, self).__init__()
        
        # Initialize weights using He initialization
        def init_weights(m):
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)  # Added BatchNorm
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)  # Added BatchNorm
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

class UNet3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, base_filters=16):
        super(UNet3D, self).__init__()
        
        # Save hyperparameters for model saving/loading
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters, dropout_p=0.1)
        self.enc2 = ConvBlock(base_filters, base_filters*2, dropout_p=0.1)
        self.enc3 = ConvBlock(base_filters*2, base_filters*4, dropout_p=0.2)
        self.enc4 = ConvBlock(base_filters*4, base_filters*8, dropout_p=0.2)
        self.enc5 = ConvBlock(base_filters*8, base_filters*16, dropout_p=0.3)
        
        # Pool
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(base_filters*16, base_filters*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_filters*16, base_filters*8, dropout_p=0.2)
        
        self.up3 = nn.ConvTranspose3d(base_filters*8, base_filters*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters*8, base_filters*4, dropout_p=0.2)
        
        self.up2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters*4, base_filters*2, dropout_p=0.1)
        
        self.up1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters*2, base_filters, dropout_p=0.1)
        
        self.final = nn.Conv3d(base_filters, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.up4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.final(dec1)
        return out  # Return logits (don't apply softmax here)
    
    def save_model(self, path, epoch=None, optimizer=None, loss=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'in_channels': self.in_channels,
                'num_classes': self.num_classes,
                'base_filters': self.base_filters
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
    
    def summary(self):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel Summary:")
        print(f"Input channels: {self.in_channels}")
        print(f"Output classes: {self.num_classes}")
        print(f"Base filters: {self.base_filters}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"\nModel Architecture:")
        print(f"{self}")

if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=3, num_classes=4).to(device)
    
    # Create test input
    x = torch.randn(1, 3, 128, 128, 128).to(device)
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    
    # Print model summary
    model.summary()
    
    # Test save and load
    model.save_model("test_model.pth")
    loaded_model, checkpoint = UNet3D.load_model("test_model.pth", device)
    print("\nModel loaded successfully!")
