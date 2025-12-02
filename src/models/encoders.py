import torch
import torch.nn as nn
import torchvision.models as models

class S1Encoder(nn.Module):
    """
    ResNet34 encoder for Sentinel-1 (SAR) data.
    Modified to accept 2 input channels (VV, VH).
    """
    def __init__(self, pretrained=True):
        super(S1Encoder, self).__init__()
        # Load standard ResNet34
        # weights="DEFAULT" is the new syntax for pretrained=True
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet34(weights=weights)
        
        # Modify the first convolution layer
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New:      Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.resnet.conv1
        
        self.resnet.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize the new layer
        # Option A: Random init (PyTorch default for new layers)
        # Option B: Average weights from RGB (better for transfer learning)
        # We'll use Option B: Average the 3 RGB weights into 2 for VV/VH
        with torch.no_grad():
            # Shape: (64, 3, 7, 7)
            original_weights = original_conv1.weight
            # Average across channel dim -> (64, 1, 7, 7)
            avg_weight = torch.mean(original_weights, dim=1, keepdim=True)
            # Duplicate to create 2 channels -> (64, 2, 7, 7)
            new_weight = avg_weight.repeat(1, 2, 1, 1)
            self.resnet.conv1.weight.copy_(new_weight)

        # Remove the Fully Connected (FC) layer and AvgPool at the end
        # We only want the feature maps for U-Net
        del self.resnet.fc
        del self.resnet.avgpool

    def forward(self, x):
        """
        Returns a list of feature maps at different scales.
        """
        features = []
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        features.append(x) # Scale 1/2 (though after maxpool it becomes 1/4, let's track carefully)
        
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        features.append(x) # Scale 1/4
        
        x = self.resnet.layer2(x)
        features.append(x) # Scale 1/8
        
        x = self.resnet.layer3(x)
        features.append(x) # Scale 1/16
        
        x = self.resnet.layer4(x)
        features.append(x) # Scale 1/32
        
        return features


class S2Encoder(nn.Module):
    """
    ResNet34 encoder for Sentinel-2 (Optical) data.
    Modified to accept 13 input channels (Multispectral).
    """
    def __init__(self, pretrained=True):
        super(S2Encoder, self).__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet34(weights=weights)
        
        # Modify first conv for 13 channels
        original_conv1 = self.resnet.conv1
        
        self.resnet.conv1 = nn.Conv2d(
            in_channels=13,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize weights
        # We keep the RGB weights for the first 3 channels (R,G,B are usually indices 3,2,1 or similar)
        # But simpler approach: Average weights again to not bias specific bands initially
        with torch.no_grad():
            original_weights = original_conv1.weight
            avg_weight = torch.mean(original_weights, dim=1, keepdim=True)
            new_weight = avg_weight.repeat(1, 13, 1, 1)
            self.resnet.conv1.weight.copy_(new_weight)
            
        del self.resnet.fc
        del self.resnet.avgpool

    def forward(self, x):
        features = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        features.append(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        features.append(x)
        x = self.resnet.layer2(x)
        features.append(x)
        x = self.resnet.layer3(x)
        features.append(x)
        x = self.resnet.layer4(x)
        features.append(x)
        return features

