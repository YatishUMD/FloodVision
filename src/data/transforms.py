import albumentations as A
import numpy as np
import os
import random
import rasterio
from albumentations.core.transforms_interface import ImageOnlyTransform

class AddCloud(ImageOnlyTransform):
    """
    Custom Augmentation: Pastes real clouds from CloudSEN12 onto the image.
    """
    def __init__(self, cloud_bank_dir="data/cloud_bank", probability=0.5, always_apply=False, p=0.5):
        super(AddCloud, self).__init__(always_apply, p)
        self.cloud_bank_dir = cloud_bank_dir
        self.probability = probability
        
        # Pre-scan cloud files
        if os.path.exists(cloud_bank_dir):
            self.cloud_files = [
                f.replace("_img.tif", "") 
                for f in os.listdir(cloud_bank_dir) 
                if f.endswith("_img.tif")
            ]
        else:
            print(f"Warning: Cloud bank not found at {cloud_bank_dir}")
            self.cloud_files = []

    def apply(self, image, **params):
        # Image is S2 (H, W, C)
        
        if not self.cloud_files:
            return image
            
        # Pick random cloud
        cloud_id = random.choice(self.cloud_files)
        img_path = os.path.join(self.cloud_bank_dir, f"{cloud_id}_img.tif")
        mask_path = os.path.join(self.cloud_bank_dir, f"{cloud_id}_mask.tif")
        
        try:
            with rasterio.open(img_path) as src:
                cloud_img = src.read() # (C, H, W)
                # Transpose to (H, W, C)
                cloud_img = np.moveaxis(cloud_img, 0, -1)
                
            with rasterio.open(mask_path) as src:
                cloud_mask = src.read(1) # (H, W)
                
            # Resize if dimensions differ (unlikely if both are 512x512)
            if cloud_img.shape[:2] != image.shape[:2]:
                # Simple crop or resize could go here
                # For now assume match or just skip
                return image
                
            # Normalization check
            # Input 'image' (S2) is usually 0-1 (float)
            # Cloud image is uint16 (0-10000+). We need to scale cloud to 0-1.
            cloud_img = cloud_img.astype(np.float32) / 10000.0
            cloud_img = np.clip(cloud_img, 0, 1)
            
            # Create blending mask
            # Cloud Mask: 0=Clear, 1=Thick, 2=Thin, 3=Shadow
            # We want to paste where mask > 0 (or just 1 & 2)
            paste_mask = (cloud_mask > 0).astype(np.float32)
            
            # Expand mask to 3 channels for broadcasting
            paste_mask = np.expand_dims(paste_mask, axis=-1)
            
            # Alpha Blending
            # Result = Cloud * Mask + Original * (1 - Mask)
            # Or simplified: Paste cloud on top
            augmented = cloud_img * paste_mask + image * (1 - paste_mask)
            
            return augmented.astype(np.float32)
            
        except Exception as e:
            # If any IO error, just return original
            return image

def get_train_transforms():
    """
    Strong augmentations for training.
    Includes: Flips, Rotations, and Real Cloud Injection.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Add Clouds with 50% probability
        AddCloud(p=0.5),
        
    ], additional_targets={'s1': 'image'})

def get_val_transforms():
    """
    No geometric augmentations for validation.
    """
    return A.Compose([
    ], additional_targets={'s1': 'image'})
