import torch
import torch.nn as nn
from src.models.encoders import S2Encoder
from src.models.decoders import DecoderBlock

class CloudNet(nn.Module):
    """
    Simplified UNet for Cloud Segmentation Pre-training.
    Uses ONLY S2Encoder + Decoder (No S1, No Fusion).
    """
    def __init__(self, num_classes=2):
        super(CloudNet, self).__init__()
        
        # Shared Encoder (This is what we want to pre-train)
        self.encoder = S2Encoder(pretrained=True)
        
        # Dimensions for ResNet34 features
        self.dims = [64, 64, 128, 256, 512]
        
        # Decoder (Similar to FloodNet but without fusion inputs)
        # Block 4: 512 + 256 -> 256
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # Block 3: 256 + 128 -> 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # Block 2: 128 + 64 -> 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # Block 1: 64 + 64 -> 64
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        # Output Head
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder Features
        features = self.encoder(x)
        f0, f1, f2, f3, f4 = features
        
        # Decode
        x = self.dec4(f4, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        
        out = self.final_conv(x)
        return out

