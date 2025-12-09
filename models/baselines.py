import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class OpticalOnlyModel(nn.Module):
    """Baseline 1: Optical-only UNet"""
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=13,  # 13 Sentinel-2 bands
            classes=1,       # Binary flood segmentation
            activation=None  # We'll use sigmoid in training
        )
    
    def forward(self, batch):
        """Forward pass using optical data only"""
        optical = batch['optical']  # (B, 13, H, W)
        return self.model(optical)


class SAROnlyModel(nn.Module):
    """Baseline 2: SAR-only UNet"""
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=2,   # 2 SAR bands (VV, VH)
            classes=1,
            activation=None
        )
    
    def forward(self, batch):
        """Forward pass using SAR data only"""
        sar = batch['sar']  # (B, 2, H, W)
        return self.model(sar)


class SimpleFusionModel(nn.Module):
    """Baseline 3: Simple early fusion (concat SAR + Optical)"""
    def __init__(self, encoder_name='resnet34', encoder_weights=None):
        super().__init__()
        # Can't use pretrained weights with 15 input channels
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=15,  # 13 optical + 2 SAR
            classes=1,
            activation=None
        )
    
    def forward(self, batch):
        """Forward pass concatenating SAR and optical"""
        sar = batch['sar']      # (B, 2, H, W)
        optical = batch['optical']  # (B, 13, H, W)
        
        # Concatenate along channel dimension
        fused = torch.cat([optical, sar], dim=1)  # (B, 15, H, W)
        return self.model(fused)


def get_model(model_type='optical', encoder_name='resnet34'):
    """
    Get baseline model
    
    Args:
        model_type: 'optical', 'sar', or 'fusion'
        encoder_name: Encoder backbone (resnet34, resnet50, efficientnet-b0, etc.)
    """
    if model_type == 'optical':
        return OpticalOnlyModel(encoder_name=encoder_name, encoder_weights='imagenet')
    elif model_type == 'sar':
        return SAROnlyModel(encoder_name=encoder_name, encoder_weights='imagenet')
    elif model_type == 'fusion':
        return SimpleFusionModel(encoder_name=encoder_name, encoder_weights=None)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
