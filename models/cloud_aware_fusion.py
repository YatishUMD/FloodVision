import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class CloudAwareFusionModel(nn.Module):
    """
    Cloud-Aware Gated Fusion Model
    
    Novel approach: Uses cloud mask to intelligently gate between SAR and Optical features
    """
    def __init__(self, encoder_name='resnet34', pretrained=False):
        super().__init__()
        
        # Create full UNet models
        self.optical_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,  # Control pretrained
            in_channels=13,
            classes=1,
            activation=None
        )
        
        self.sar_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,  # Control pretrained
            in_channels=2,
            classes=1,
            activation=None
        )
        
        # Cloud mask processing network
        self.cloud_processor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global pooling
        )
        
        # Gating mechanism - produces a scalar weight per sample
        self.gate_network = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1]: 0=optical, 1=SAR
        )
    
    def forward(self, batch):
        optical = batch['optical']  # (B, 13, H, W)
        sar = batch['sar']          # (B, 2, H, W)
        cloud_mask = batch['cloud_mask']  # (B, 1, H, W)
        
        # Get predictions from both modalities
        optical_pred = self.optical_unet(optical)  # (B, 1, H, W)
        sar_pred = self.sar_unet(sar)              # (B, 1, H, W)
        
        # Process cloud mask to get gating weight
        cloud_features = self.cloud_processor(cloud_mask)  # (B, 32, 1, 1)
        cloud_features = cloud_features.view(cloud_features.size(0), -1)  # (B, 32)
        
        # Generate per-sample gating weights
        gate = self.gate_network(cloud_features)  # (B, 1)
        gate = gate.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting
        
        # Weighted fusion based on cloud coverage
        # High cloud (gate→1) → use more SAR
        # Clear sky (gate→0) → use more optical
        fused_pred = (1 - gate) * optical_pred + gate * sar_pred
        
        return fused_pred, gate.squeeze()  # Return prediction and gating weights


def get_cloud_aware_model(encoder_name='resnet34', pretrained=False):
    return CloudAwareFusionModel(encoder_name=encoder_name, pretrained=pretrained)
