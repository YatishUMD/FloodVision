import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from src.models.encoders import S1Encoder, S2Encoder
from src.models.fusion import GatedFusion

class DecoderBlock(nn.Module):
    """
    Standard UNet Decoder Block
    Upsample -> Concat Skip Connection -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 1. Upsampling layer (Bilinear is simple and effective)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. Convolution after concatenation
        # Input channels = in_channels (from below) + skip_channels (from encoder)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle potential padding issues if dimensions don't match exactly
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class FloodNet(nn.Module):
    """
    Full Network Architecture:
    S1 Encoder + S2 Encoder -> Gated Fusion at each scale -> UNet Decoder
    """
    def __init__(self, num_classes=2, s2_weights_path=None):
        super(FloodNet, self).__init__()
        
        # 1. Encoders
        self.s1_encoder = S1Encoder(pretrained=True)
        self.s2_encoder = S2Encoder(pretrained=True)
        
        # Load pre-trained cloud weights if provided
        if s2_weights_path and os.path.exists(s2_weights_path):
            print(f"Loading pre-trained S2 weights from {s2_weights_path}")
            # The saved state_dict is from model.encoder (S2Encoder)
            self.s2_encoder.load_state_dict(torch.load(s2_weights_path))
        
        # Channel counts for ResNet34 features at each scale
        # [Scale 1/2, Scale 1/4, Scale 1/8, Scale 1/16, Scale 1/32]
        # [64, 64, 128, 256, 512]
        self.dims = [64, 64, 128, 256, 512]
        
        # 2. Fusion Modules
        # We need a fusion module for EACH scale
        self.fusion0 = GatedFusion(self.dims[0])
        self.fusion1 = GatedFusion(self.dims[1])
        self.fusion2 = GatedFusion(self.dims[2])
        self.fusion3 = GatedFusion(self.dims[3])
        self.fusion4 = GatedFusion(self.dims[4])
        
        # 3. Decoder Blocks
        # Moving from bottom (Scale 1/32) up to full resolution
        
        # Block 4: Takes Fusion4 (512) and Fusion3 (256) -> Outputs 256
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        
        # Block 3: Takes (256) and Fusion2 (128) -> Outputs 128
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        
        # Block 2: Takes (128) and Fusion1 (64) -> Outputs 64
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        
        # Block 1: Takes (64) and Fusion0 (64) -> Outputs 64
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        # 4. Final Output Head
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # Restore from 1/2 to Full
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1) # Logits
        )

    def forward(self, s1, s2):
        # 1. Encode
        f1 = self.s1_encoder(s1) # List of 5 features
        f2 = self.s2_encoder(s2) # List of 5 features
        
        # 2. Fuse Features at each scale
        x0 = self.fusion0(f1[0], f2[0]) # 64 ch
        x1 = self.fusion1(f1[1], f2[1]) # 64 ch
        x2 = self.fusion2(f1[2], f2[2]) # 128 ch
        x3 = self.fusion3(f1[3], f2[3]) # 256 ch
        x4 = self.fusion4(f1[4], f2[4]) # 512 ch (Bottleneck)
        
        # 3. Decode
        # Up from x4 (bottleneck) using x3 as skip
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        
        # 4. Final Prediction
        out = self.final_conv(d1)
        
        return out

