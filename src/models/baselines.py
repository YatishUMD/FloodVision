import torch
import torch.nn as nn
from src.models.encoders import S1Encoder, S2Encoder
from src.models.decoders import DecoderBlock

class OpticalUNet(nn.Module):
    """
    Baseline: Optical-only U-Net (Sentinel-2)
    """
    def __init__(self, num_classes=2):
        super(OpticalUNet, self).__init__()
        self.encoder = S2Encoder(pretrained=True)
        # dims: [64, 64, 128, 256, 512]
        
        # Same decoder structure as FloodNet but without fusion inputs
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, s1, s2):
        # Ignores s1, uses s2
        features = self.encoder(s2)
        f0, f1, f2, f3, f4 = features
        
        x = self.dec4(f4, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        
        return self.final_conv(x)

class SarUNet(nn.Module):
    """
    Baseline: SAR-only U-Net (Sentinel-1)
    """
    def __init__(self, num_classes=2):
        super(SarUNet, self).__init__()
        self.encoder = S1Encoder(pretrained=True)
        
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, s1, s2):
        # Ignores s2, uses s1
        features = self.encoder(s1)
        f0, f1, f2, f3, f4 = features
        
        x = self.dec4(f4, f3)
        x = self.dec3(x, f2)
        x = self.dec2(x, f1)
        x = self.dec1(x, f0)
        
        return self.final_conv(x)

class SimpleFusionUNet(nn.Module):
    """
    Baseline: Simple Early Fusion (Concatenation)
    concatenates features at each scale.
    """
    def __init__(self, num_classes=2):
        super(SimpleFusionUNet, self).__init__()
        self.s1_encoder = S1Encoder(pretrained=True)
        self.s2_encoder = S2Encoder(pretrained=True)
        
        # Concatenated dimensions
        # f4: 512+512 = 1024
        # f3: 256+256 = 512
        # f2: 128+128 = 256
        # f1: 64+64 = 128
        # f0: 64+64 = 128
        
        self.dec4 = DecoderBlock(in_channels=1024, skip_channels=512, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=256, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=128, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=128, out_channels=64)
        
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, s1, s2):
        f1_list = self.s1_encoder(s1)
        f2_list = self.s2_encoder(s2)
        
        # Concatenate at each scale
        x0 = torch.cat([f1_list[0], f2_list[0]], dim=1)
        x1 = torch.cat([f1_list[1], f2_list[1]], dim=1)
        x2 = torch.cat([f1_list[2], f2_list[2]], dim=1)
        x3 = torch.cat([f1_list[3], f2_list[3]], dim=1)
        x4 = torch.cat([f1_list[4], f2_list[4]], dim=1)
        
        x = self.dec4(x4, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, x0)
        
        return self.final_conv(x)
