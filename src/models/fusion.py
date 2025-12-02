import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Gated Fusion Module.
    Learns to weigh Sentinel-1 and Sentinel-2 features adaptively.
    
    Logic:
    1. Concatenate S1 and S2 features.
    2. Pass through a small conv network to generate an Attention Map (A).
    3. Fused Output = A * S2 + (1 - A) * S1
    
    If S2 (Optical) is cloudy, the network should learn to make A -> 0,
    forcing reliance on S1 (Radar).
    """
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        
        # Attention generation block
        # Input: S1 + S2 concatenated (2 * channels)
        # Output: 1 channel attention map (0 to 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, s1_feat, s2_feat):
        """
        Args:
            s1_feat: Feature map from SAR Encoder (B, C, H, W)
            s2_feat: Feature map from Optical Encoder (B, C, H, W)
        Returns:
            fused: Weighted combination (B, C, H, W)
        """
        # Concatenate along channel dimension
        cat = torch.cat([s1_feat, s2_feat], dim=1)
        
        # Compute attention map (B, 1, H, W)
        # 1 means "Trust Optical", 0 means "Trust Radar"
        attention = self.gate(cat)
        
        # Weighted sum
        # Broadcast attention map across all channels
        fused = attention * s2_feat + (1 - attention) * s1_feat
        
        return fused

