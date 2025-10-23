"""
Dataset Loaders for Flood Vision Project
Handles loading and batching of SEN12MS-CR, SEN12-FLOOD, and SEN1Floods11 datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
import glob
from data_preprocessing import create_preprocessing_pipeline

logger = logging.getLogger(__name__)

class FloodDataset(Dataset):
    """
    Base dataset class for flood-related satellite imagery
    """
    
    def __init__(self, data_path: str, split: str = 'train', 
                 config: Optional[Dict] = None, transform: Optional[callable] = None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to the dataset
            split: Dataset split ('train', 'val', 'test')
            config: Configuration dictionary
            transform: Optional transform function
        """
        self.data_path = Path(data_path)
        self.split = split
        self.config = config or {}
        self.transform = transform
        
        # Initialize preprocessing pipeline
        self.preprocessing_pipeline = create_preprocessing_pipeline(self.config)
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        """Load sample metadata - to be implemented by subclasses"""
        raise NotImplementedError
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load data
        sar_data = self._load_sar_data(sample['sar_path'])
        optical_data = self._load_optical_data(sample['optical_path'])
        
        # Load mask if available
        mask_data = None
        if 'mask_path' in sample and sample['mask_path']:
            mask_data = self._load_mask_data(sample['mask_path'])
        
        # Preprocess data
        sar_data = self.preprocessing_pipeline['preprocessor'].preprocess_sar(sar_data)
        optical_data = self.preprocessing_pipeline['preprocessor'].preprocess_optical(optical_data)
        
        # Apply augmentation for training
        if self.split == 'train' and self.config.get('augmentation', True):
            sar_data, optical_data, mask_data = self.preprocessing_pipeline['augmentation'].augment_data(
                sar_data, optical_data, mask_data
            )
        
        # Align data
        sar_data, optical_data = self.preprocessing_pipeline['alignment'].align_spatial_extent(
            sar_data, optical_data
        )
        
        # Create sample dictionary
        sample_dict = {
            'sar': torch.from_numpy(sar_data).float(),
            'optical': torch.from_numpy(optical_data).float(),
            'sample_id': sample.get('sample_id', idx)
        }
        
        if mask_data is not None:
            sample_dict['mask'] = torch.from_numpy(mask_data).long()
        
        # Apply transform if provided
        if self.transform:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict
    
    def _load_sar_data(self, sar_path: str) -> np.ndarray:
        """Load SAR data from file"""
        try:
            with rasterio.open(sar_path) as src:
                return src.read()  # Shape: (bands, height, width)
        except Exception as e:
            logger.error(f"Error loading SAR data from {sar_path}: {e}")
            raise
    
    def _load_optical_data(self, optical_path: str) -> np.ndarray:
        """Load optical data from file"""
        try:
            with rasterio.open(optical_path) as src:
                return src.read()  # Shape: (bands, height, width)
        except Exception as e:
            logger.error(f"Error loading optical data from {optical_path}: {e}")
            raise
    
    def _load_mask_data(self, mask_path: str) -> np.ndarray:
        """Load mask data from file"""
        try:
            with rasterio.open(mask_path) as src:
                return src.read(1)  # Shape: (height, width)
        except Exception as e:
            logger.error(f"Error loading mask data from {mask_path}: {e}")
            return None

class SEN12MSDataset(FloodDataset):
    """
    Dataset loader for SEN12MS-CR dataset
    """
    
    def _load_samples(self) -> List[Dict]:
        """Load SEN12MS-CR samples"""
        samples = []
        
        # Look for SEN12MS-CR structure
        sar_dir = self.data_path / "sar"
        optical_dir = self.data_path / "optical"
        mask_dir = self.data_path / "masks"
        
        if not sar_dir.exists() or not optical_dir.exists():
            logger.error(f"SEN12MS-CR directories not found in {self.data_path}")
            return samples
        
        # Find matching SAR and optical files
        sar_files = sorted(list(sar_dir.glob("*.tif")))
        optical_files = sorted(list(optical_dir.glob("*.tif")))
        
        # Match files by name
        sar_dict = {f.stem: f for f in sar_files}
        optical_dict = {f.stem: f for f in optical_files}
        
        for sample_id in sar_dict.keys():
            if sample_id in optical_dict:
                sample = {
                    'sample_id': sample_id,
                    'sar_path': str(sar_dict[sample_id]),
                    'optical_path': str(optical_dict[sample_id])
                }
                
                # Add mask if available
                if mask_dir.exists():
                    mask_file = mask_dir / f"{sample_id}.tif"
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)
                
                samples.append(sample)
        
        return samples

class SEN12FloodDataset(FloodDataset):
    """
    Dataset loader for SEN12-FLOOD dataset
    """
    
    def _load_samples(self) -> List[Dict]:
        """Load SEN12-FLOOD samples"""
        samples = []
        
        # Look for SEN12-FLOOD structure
        sar_dir = self.data_path / "sar"
        optical_dir = self.data_path / "optical"
        flood_mask_dir = self.data_path / "flood_masks"
        
        if not sar_dir.exists() or not optical_dir.exists():
            logger.error(f"SEN12-FLOOD directories not found in {self.data_path}")
            return samples
        
        # Find matching files
        sar_files = sorted(list(sar_dir.glob("*.tif")))
        optical_files = sorted(list(optical_dir.glob("*.tif")))
        
        # Match files by name
        sar_dict = {f.stem: f for f in sar_files}
        optical_dict = {f.stem: f for f in optical_files}
        
        for sample_id in sar_dict.keys():
            if sample_id in optical_dict:
                sample = {
                    'sample_id': sample_id,
                    'sar_path': str(sar_dict[sample_id]),
                    'optical_path': str(optical_dict[sample_id])
                }
                
                # Add flood mask if available
                if flood_mask_dir.exists():
                    mask_file = flood_mask_dir / f"{sample_id}.tif"
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)
                
                samples.append(sample)
        
        return samples

class SEN1Floods11Dataset(FloodDataset):
    """
    Dataset loader for SEN1Floods11 dataset
    """
    
    def _load_samples(self) -> List[Dict]:
        """Load SEN1Floods11 samples"""
        samples = []
        
        # Look for SEN1Floods11 structure
        sar_dir = self.data_path / "sar"
        optical_dir = self.data_path / "optical"
        flood_mask_dir = self.data_path / "flood_masks"
        
        if not sar_dir.exists() or not optical_dir.exists():
            logger.error(f"SEN1Floods11 directories not found in {self.data_path}")
            return samples
        
        # Find matching files
        sar_files = sorted(list(sar_dir.glob("*.tif")))
        optical_files = sorted(list(optical_dir.glob("*.tif")))
        
        # Match files by name
        sar_dict = {f.stem: f for f in sar_files}
        optical_dict = {f.stem: f for f in optical_files}
        
        for sample_id in sar_dict.keys():
            if sample_id in optical_dict:
                sample = {
                    'sample_id': sample_id,
                    'sar_path': str(sar_dict[sample_id]),
                    'optical_path': str(optical_dict[sample_id])
                }
                
                # Add flood mask if available
                if flood_mask_dir.exists():
                    mask_file = flood_mask_dir / f"{sample_id}.tif"
                    if mask_file.exists():
                        sample['mask_path'] = str(mask_file)
                
                samples.append(sample)
        
        return samples

class CombinedFloodDataset(Dataset):
    """
    Combined dataset that merges multiple flood datasets
    """
    
    def __init__(self, datasets: List[FloodDataset]):
        """
        Initialize combined dataset
        
        Args:
            datasets: List of FloodDataset instances
        """
        self.datasets = datasets
        self.total_samples = sum(len(dataset) for dataset in datasets)
        
        # Create mapping from global index to dataset and local index
        self.index_mapping = []
        for dataset_idx, dataset in enumerate(datasets):
            for local_idx in range(len(dataset)):
                self.index_mapping.append((dataset_idx, local_idx))
        
        logger.info(f"Combined dataset with {self.total_samples} total samples")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample from combined dataset"""
        dataset_idx, local_idx = self.index_mapping[idx]
        return self.datasets[dataset_idx][local_idx]

def create_data_loaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Create data loaders for all datasets and splits
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing data loaders
    """
    data_loaders = {}
    
    # Dataset configurations
    dataset_configs = {
        'sen12ms_cr': {
            'class': SEN12MSDataset,
            'path': config['datasets']['sen12ms_cr']['path']
        },
        'sen12_flood': {
            'class': SEN12FloodDataset,
            'path': config['datasets']['sen12_flood']['path']
        },
        'sen1floods11': {
            'class': SEN1Floods11Dataset,
            'path': config['datasets']['sen1floods11']['path']
        }
    }
    
    # Create datasets for each split
    for split in ['train', 'val', 'test']:
        datasets = []
        
        for dataset_name, dataset_config in dataset_configs.items():
            dataset_path = dataset_config['path']
            
            if os.path.exists(dataset_path):
                dataset_class = dataset_config['class']
                dataset = dataset_class(
                    data_path=dataset_path,
                    split=split,
                    config=config
                )
                
                if len(dataset) > 0:
                    datasets.append(dataset)
                    logger.info(f"Created {dataset_name} {split} dataset with {len(dataset)} samples")
        
        # Combine datasets if multiple exist
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        elif len(datasets) > 1:
            combined_dataset = CombinedFloodDataset(datasets)
        else:
            logger.warning(f"No datasets found for {split} split")
            continue
        
        # Create data loader
        batch_size = config.get('batch_size', 8)
        num_workers = config.get('num_workers', 4)
        shuffle = (split == 'train')
        
        data_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        data_loaders[split] = data_loader
    
    return data_loaders

def create_single_dataset_loader(dataset_name: str, split: str, config: Dict) -> DataLoader:
    """
    Create a data loader for a single dataset
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split
        config: Configuration dictionary
        
    Returns:
        DataLoader instance
    """
    dataset_configs = {
        'sen12ms_cr': SEN12MSDataset,
        'sen12_flood': SEN12FloodDataset,
        'sen1floods11': SEN1Floods11Dataset
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_path = config['datasets'][dataset_name]['path']
    dataset_class = dataset_configs[dataset_name]
    
    dataset = dataset_class(
        data_path=dataset_path,
        split=split,
        config=config
    )
    
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)
    shuffle = (split == 'train')
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return data_loader

def get_dataset_statistics(data_loader: DataLoader) -> Dict:
    """
    Compute dataset statistics for normalization
    
    Args:
        data_loader: DataLoader instance
        
    Returns:
        Dictionary containing mean and std for SAR and optical data
    """
    sar_data_list = []
    optical_data_list = []
    
    logger.info("Computing dataset statistics...")
    
    for batch in data_loader:
        sar_data_list.append(batch['sar'])
        optical_data_list.append(batch['optical'])
        
        # Limit to first few batches for efficiency
        if len(sar_data_list) >= 10:
            break
    
    # Concatenate all data
    sar_data = torch.cat(sar_data_list, dim=0)
    optical_data = torch.cat(optical_data_list, dim=0)
    
    # Compute statistics
    sar_mean = torch.mean(sar_data, dim=(0, 2, 3))
    sar_std = torch.std(sar_data, dim=(0, 2, 3))
    
    optical_mean = torch.mean(optical_data, dim=(0, 2, 3))
    optical_std = torch.std(optical_data, dim=(0, 2, 3))
    
    statistics = {
        'sar': {
            'mean': sar_mean.numpy(),
            'std': sar_std.numpy()
        },
        'optical': {
            'mean': optical_mean.numpy(),
            'std': optical_std.numpy()
        }
    }
    
    logger.info("Dataset statistics computed successfully")
    return statistics

# Example usage
if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create data loaders
    data_loaders = create_data_loaders(config)
    
    # Test data loading
    for split, data_loader in data_loaders.items():
        print(f"\n{split.upper()} DataLoader:")
        print(f"Number of batches: {len(data_loader)}")
        
        # Get a sample batch
        for batch in data_loader:
            print(f"SAR shape: {batch['sar'].shape}")
            print(f"Optical shape: {batch['optical'].shape}")
            if 'mask' in batch:
                print(f"Mask shape: {batch['mask'].shape}")
            break
