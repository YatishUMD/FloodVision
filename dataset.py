import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SEN1Floods11Dataset(Dataset):
    def __init__(self, data_dir, split='train', augment=False, use_cloud_mask=True):
        """
        SEN1Floods11 Dataset for flood segmentation
        
        Args:
            data_dir: Path to sen1floods11 folder
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            use_cloud_mask: Whether to generate cloud masks
        """
        self.data_dir = Path(data_dir)
        self.augment = augment and (split == 'train')
        self.use_cloud_mask = use_cloud_mask
        
        # Paths to data
        handlabeled = self.data_dir / "data/flood_events/HandLabeled"
        self.s1_dir = handlabeled / "S1Hand"
        self.s2_dir = handlabeled / "S2Hand"
        self.label_dir = handlabeled / "LabelHand"
        
        # Verify directories exist
        assert self.s1_dir.exists(), f"S1Hand not found: {self.s1_dir}"
        assert self.s2_dir.exists(), f"S2Hand not found: {self.s2_dir}"
        assert self.label_dir.exists(), f"LabelHand not found: {self.label_dir}"
        
        # Get all samples
        self.samples = sorted([f.stem.replace('_S1Hand', '') 
                              for f in self.s1_dir.glob("*_S1Hand.tif")])
        
        # Split data (70/15/15)
        n = len(self.samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if split == 'train':
            self.samples = self.samples[:train_end]
        elif split == 'val':
            self.samples = self.samples[train_end:val_end]
        else:  # test
            self.samples = self.samples[val_end:]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def load_tif(self, filepath):
        """Load a GeoTIFF file"""
        with rasterio.open(filepath) as src:
            data = src.read()
        return data.astype(np.float32)
    
    def generate_cloud_mask(self, s2_data):
        """Generate cloud mask using Sentinel-2 cirrus band"""
        cirrus = s2_data[10, :, :]  # Band 10 is cirrus
        cloud_mask = (cirrus > 1000).astype(np.float32)
        return cloud_mask
    
    def augment_data(self, s1, s2, label, cloud_mask):
        """Simple augmentation: random flips and rotations"""
        import random
        
        # Random horizontal flip
        if random.random() > 0.5:
            s1 = np.flip(s1, axis=2).copy()
            s2 = np.flip(s2, axis=2).copy()
            label = np.flip(label, axis=1).copy()
            cloud_mask = np.flip(cloud_mask, axis=1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            s1 = np.flip(s1, axis=1).copy()
            s2 = np.flip(s2, axis=1).copy()
            label = np.flip(label, axis=0).copy()
            cloud_mask = np.flip(cloud_mask, axis=0).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            s1 = np.rot90(s1, k, axes=(1, 2)).copy()
            s2 = np.rot90(s2, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            cloud_mask = np.rot90(cloud_mask, k, axes=(0, 1)).copy()
        
        return s1, s2, label, cloud_mask
    
    def __getitem__(self, idx):
        sample_name = self.samples[idx]
    
        # Load data
        s1 = self.load_tif(self.s1_dir / f"{sample_name}_S1Hand.tif")
        s2 = self.load_tif(self.s2_dir / f"{sample_name}_S2Hand.tif")
        label = self.load_tif(self.label_dir / f"{sample_name}_LabelHand.tif")
     
        # Normalize S2 (TOA reflectance scaled by 10000)
        s2 = s2.astype(np.float32)
        s2 = s2 / 10000.0
        s2 = np.clip(s2, 0, 1)
        s2 = np.nan_to_num(s2, nan=0.0, posinf=1.0, neginf=0.0)
    
        # Normalize S1 (SAR in dB)
        s1 = s1.astype(np.float32)
        # Replace invalid values first
        s1 = np.nan_to_num(s1, nan=-50.0, posinf=10.0, neginf=-50.0)
        # Clip to typical SAR range (-50 to +10 dB)
        s1 = np.clip(s1, -50, 10)
        # Normalize to [0, 1]
        s1 = (s1 + 50.0) / 60.0
    
        # Process label (-1: invalid, 0: land, 1: water)
        label = np.where(label == -1, 0, label).astype(np.float32)
        label = label[0] if label.ndim == 3 else label
    
        # Generate cloud mask
        if self.use_cloud_mask:
            cloud_mask = self.generate_cloud_mask(s2)
        else:
            cloud_mask = np.zeros((s2.shape[1], s2.shape[2]), dtype=np.float32)
    
        # Apply augmentation if training
        if self.augment:
            s1, s2, label, cloud_mask = self.augment_data(s1, s2, label, cloud_mask)
    
        # Convert to tensors
        s1 = torch.FloatTensor(s1)
        s2 = torch.FloatTensor(s2)
        label = torch.FloatTensor(label).unsqueeze(0)
        cloud_mask = torch.FloatTensor(cloud_mask).unsqueeze(0)
    
        return {
            'sar': s1,           # (2, H, W)
            'optical': s2,       # (13, H, W)
            'label': label,      # (1, H, W)
            'cloud_mask': cloud_mask,  # (1, H, W)
            'name': sample_name
        }

def get_dataloaders(data_dir, batch_size=8, num_workers=4):
    """Create train/val/test dataloaders"""
    train_dataset = SEN1Floods11Dataset(data_dir=data_dir, split='train', augment=True)
    val_dataset = SEN1Floods11Dataset(data_dir=data_dir, split='val', augment=False)
    test_dataset = SEN1Floods11Dataset(data_dir=data_dir, split='test', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
