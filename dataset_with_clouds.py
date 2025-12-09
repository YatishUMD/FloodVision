import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from cloud_augmentation import augment_with_clouds

class SEN1Floods11CloudDataset(Dataset):
    """Dataset with cloud augmentation during training"""
    
    def __init__(self, data_dir, split='train', augment=False, 
                 use_cloud_mask=True, cloud_augmentation=True):
        self.data_dir = Path(data_dir)
        self.augment = augment and (split == 'train')
        self.use_cloud_mask = use_cloud_mask
        self.cloud_augmentation = cloud_augmentation and (split == 'train')
        
        # Paths
        handlabeled = self.data_dir / "data/flood_events/HandLabeled"
        self.s1_dir = handlabeled / "S1Hand"
        self.s2_dir = handlabeled / "S2Hand"
        self.label_dir = handlabeled / "LabelHand"
        
        # Get samples
        self.samples = sorted([f.stem.replace('_S1Hand', '') 
                              for f in self.s1_dir.glob("*_S1Hand.tif")])
        
        # Split data
        n = len(self.samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if split == 'train':
            self.samples = self.samples[:train_end]
        elif split == 'val':
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        if self.cloud_augmentation:
            print(f"  → Cloud augmentation ENABLED (50% probability)")
    
    def __len__(self):
        return len(self.samples)
    
    def load_tif(self, filepath):
        with rasterio.open(filepath) as src:
            data = src.read()
        return data.astype(np.float32)
    
    def generate_cloud_mask(self, s2_data):
        cirrus = s2_data[10, :, :]
        cloud_mask = (cirrus > 1000).astype(np.float32)
        return cloud_mask
    
    def augment_data(self, s1, s2, label, cloud_mask):
        """Geometric augmentation"""
        import random
        
        if random.random() > 0.5:
            s1 = np.flip(s1, axis=2).copy()
            s2 = np.flip(s2, axis=2).copy()
            label = np.flip(label, axis=1).copy()
            cloud_mask = np.flip(cloud_mask, axis=1).copy()
        
        if random.random() > 0.5:
            s1 = np.flip(s1, axis=1).copy()
            s2 = np.flip(s2, axis=1).copy()
            label = np.flip(label, axis=0).copy()
            cloud_mask = np.flip(cloud_mask, axis=0).copy()
        
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
        
        # Normalize
        s1 = s1.astype(np.float32)
        s1 = np.nan_to_num(s1, nan=-50.0, posinf=10.0, neginf=-50.0)
        s1 = np.clip(s1, -50, 10)
        s1 = (s1 + 50.0) / 60.0
        
        s2 = s2.astype(np.float32)
        s2 = s2 / 10000.0
        s2 = np.clip(s2, 0, 1)
        s2 = np.nan_to_num(s2, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Process label
        label = np.where(label == -1, 0, label).astype(np.float32)
        label = label[0] if label.ndim == 3 else label
        
        # Generate original cloud mask
        if self.use_cloud_mask:
            cloud_mask = self.generate_cloud_mask(s2)
        else:
            cloud_mask = np.zeros((s2.shape[1], s2.shape[2]), dtype=np.float32)
        
        # CLOUD AUGMENTATION 
        if self.cloud_augmentation:
            s2, s1, label, cloud_mask = augment_with_clouds(
                s2, s1, label, cloud_mask,
                cloud_probability=0.5,
                coverage_range=(0.3, 0.7)
            )
        
        # Apply geometric augmentation
        if self.augment:
            s1, s2, label, cloud_mask = self.augment_data(s1, s2, label, cloud_mask)
        
        # Convert to tensors
        s1 = torch.FloatTensor(s1)
        s2 = torch.FloatTensor(s2)
        label = torch.FloatTensor(label).unsqueeze(0)
        cloud_mask = torch.FloatTensor(cloud_mask).unsqueeze(0)
        
        return {
            'sar': s1,
            'optical': s2,
            'label': label,
            'cloud_mask': cloud_mask,
            'name': sample_name
        }


def get_cloud_dataloaders(data_dir, batch_size=8, num_workers=4, cloud_augmentation=True):
    """Create dataloaders with cloud augmentation"""
    train_dataset = SEN1Floods11CloudDataset(
        data_dir=data_dir, 
        split='train', 
        augment=True,
        cloud_augmentation=cloud_augmentation
    )
    
    val_dataset = SEN1Floods11CloudDataset(
        data_dir=data_dir,
        split='val',
        augment=False,
        cloud_augmentation=False  # No augmentation for validation
    )
    
    test_dataset = SEN1Floods11CloudDataset(
        data_dir=data_dir,
        split='test',
        augment=False,
        cloud_augmentation=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
