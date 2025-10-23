"""
ETL Pipeline for Flood Vision Project
Handles extraction, transformation, and loading of Sentinel-1/2 imagery datasets
"""

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import cv2
import pandas as pd
from datetime import datetime
import patchify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloodETLPipeline:
    """
    Main ETL pipeline for processing flood-related satellite imagery datasets
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the ETL pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for the ETL pipeline"""
        return {
            "datasets": {
                "sentinel_data": {
                    "path": "./data/sentinel_data",
                    "sar_bands": ["VV", "VH"],
                    "optical_bands": ["B2", "B3", "B4", "B8"],
                    "source": "Copernicus Data Space Ecosystem",
                    "cloud_coverage_filter": "<20%",
                    "time_period": "2021-2022"
                },
                "sen12_flood": {
                    "path": "./data/SEN12-FLOOD",
                    "sar_bands": ["VV", "VH"],
                    "optical_bands": ["B2", "B3", "B4", "B8"]
                },
                "sen1floods11": {
                    "path": "./data/SEN1Floods11",
                    "sar_bands": ["VV", "VH"],
                    "optical_bands": ["B2", "B3", "B4", "B8"]
                }
            },
            "processing": {
                "target_resolution": 10,  # meters
                "patch_size": 256,
                "overlap": 0.5,  # 50% overlap as mentioned in progress report
                "normalization": "zscore",  # z-score normalization as mentioned
                "augmentation": True
            },
            "output": {
                "processed_path": "./data/processed",
                "metadata_csv_path": "./data/metadata.csv",
                "train_split": 0.7,
                "val_split": 0.2,
                "test_split": 0.1
            }
        }
    
    def setup_directories(self):
        """Create necessary directories for data processing"""
        dirs_to_create = [
            self.config["output"]["processed_path"],
            os.path.join(self.config["output"]["processed_path"], "train"),
            os.path.join(self.config["output"]["processed_path"], "val"),
            os.path.join(self.config["output"]["processed_path"], "test"),
            os.path.join(self.config["output"]["processed_path"], "metadata")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def extract_sar_data(self, sar_path: str) -> np.ndarray:
        """Extract and preprocess SAR data from Sentinel-1"""
        try:
            with rasterio.open(sar_path) as src:
                # Read VV and VH bands
                sar_data = src.read()  # Shape: (bands, height, width)
                
                # Convert to dB scale
                sar_data = 10 * np.log10(np.maximum(sar_data, 1e-10))
                
                # Normalize to [0, 1] range
                sar_data = self._normalize_sar(sar_data)
                
                return sar_data
                
        except Exception as e:
            logger.error(f"Error extracting SAR data from {sar_path}: {e}")
            return None
    
    def extract_optical_data(self, optical_path: str) -> np.ndarray:
        """Extract and preprocess optical data from Sentinel-2"""
        try:
            with rasterio.open(optical_path) as src:
                optical_data = src.read()  # Shape: (bands, height, width)
                
                # Normalize to [0, 1] range
                optical_data = optical_data.astype(np.float32) / 10000.0
                
                # Clip extreme values
                optical_data = np.clip(optical_data, 0, 1)
                
                return optical_data
                
        except Exception as e:
            logger.error(f"Error extracting optical data from {optical_path}: {e}")
            return None
    
    def _normalize_sar(self, sar_data: np.ndarray) -> np.ndarray:
        """Normalize SAR data to [0, 1] range"""
        # Clip extreme values
        sar_data = np.clip(sar_data, -30, 5)  # Typical SAR range
        
        # Min-max normalization
        min_val = np.min(sar_data)
        max_val = np.max(sar_data)
        
        if max_val > min_val:
            sar_data = (sar_data - min_val) / (max_val - min_val)
        
        return sar_data
    
    def align_imagery(self, sar_data: np.ndarray, optical_data: np.ndarray, 
                     sar_meta: Dict, optical_meta: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Align SAR and optical imagery to same spatial resolution and extent"""
        
        # Get target resolution from config
        target_resolution = self.config["processing"]["target_resolution"]
        
        # Determine which dataset has higher resolution
        sar_res = sar_meta.get('transform', (0, 10, 0, 0, 0, -10))[1]
        optical_res = optical_meta.get('transform', (0, 10, 0, 0, 0, -10))[1]
        
        # Resample to target resolution
        if abs(sar_res) != target_resolution:
            sar_data = self._resample_data(sar_data, sar_meta, target_resolution)
        
        if abs(optical_res) != target_resolution:
            optical_data = self._resample_data(optical_data, optical_meta, target_resolution)
        
        # Ensure same spatial extent
        sar_data, optical_data = self._match_extents(sar_data, optical_data)
        
        return sar_data, optical_data
    
    def _resample_data(self, data: np.ndarray, metadata: Dict, target_resolution: float) -> np.ndarray:
        """Resample data to target resolution"""
        # This is a simplified resampling - in practice, you'd use rasterio's reproject
        current_res = abs(metadata.get('transform', (0, 10, 0, 0, 0, -10))[1])
        scale_factor = current_res / target_resolution
        
        if scale_factor != 1.0:
            new_height = int(data.shape[1] * scale_factor)
            new_width = int(data.shape[2] * scale_factor)
            
            resampled_data = np.zeros((data.shape[0], new_height, new_width), dtype=data.dtype)
            
            for i in range(data.shape[0]):
                resampled_data[i] = cv2.resize(
                    data[i], 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            return resampled_data
        
        return data
    
    def _match_extents(self, sar_data: np.ndarray, optical_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Match spatial extents of SAR and optical data"""
        # Find minimum dimensions
        min_height = min(sar_data.shape[1], optical_data.shape[1])
        min_width = min(sar_data.shape[2], optical_data.shape[2])
        
        # Crop both datasets to minimum dimensions
        sar_cropped = sar_data[:, :min_height, :min_width]
        optical_cropped = optical_data[:, :min_height, :min_width]
        
        return sar_cropped, optical_cropped
    
    def create_patches(self, sar_data: np.ndarray, optical_data: np.ndarray, 
                      mask_data: Optional[np.ndarray] = None) -> List[Dict]:
        """Create patches from full images with 50% overlap"""
        patch_size = self.config["processing"]["patch_size"]
        overlap = self.config["processing"]["overlap"]
        step_size = int(patch_size * (1 - overlap))
        
        patches = []
        
        height, width = sar_data.shape[1], sar_data.shape[2]
        
        for i in range(0, height - patch_size + 1, step_size):
            for j in range(0, width - patch_size + 1, step_size):
                patch_data = {
                    'sar': sar_data[:, i:i+patch_size, j:j+patch_size],
                    'optical': optical_data[:, i:i+patch_size, j:j+patch_size],
                    'coordinates': (i, j),
                    'patch_id': f"{i}_{j}"
                }
                
                if mask_data is not None:
                    patch_data['mask'] = mask_data[i:i+patch_size, j:j+patch_size]
                
                patches.append(patch_data)
        
        return patches
    
    def process_dataset(self, dataset_name: str) -> bool:
        """Process a complete dataset"""
        logger.info(f"Processing dataset: {dataset_name}")
        
        dataset_config = self.config["datasets"].get(dataset_name)
        if not dataset_config:
            logger.error(f"Dataset {dataset_name} not found in config")
            return False
        
        dataset_path = dataset_config["path"]
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path {dataset_path} does not exist")
            return False
        
        # Find all image pairs in the dataset
        image_pairs = self._find_image_pairs(dataset_path)
        
        # Initialize metadata list
        metadata_list = []
        processed_count = 0
        
        for sar_path, optical_path, mask_path in image_pairs:
            try:
                # Extract data
                sar_data = self.extract_sar_data(sar_path)
                optical_data = self.extract_optical_data(optical_path)
                
                if sar_data is None or optical_data is None:
                    continue
                
                # Load mask if available
                mask_data = None
                if mask_path and os.path.exists(mask_path):
                    mask_data = self._load_mask(mask_path)
                
                # Align imagery
                sar_data, optical_data = self.align_imagery(sar_data, optical_data, {}, {})
                
                # Create patches
                patches = self.create_patches(sar_data, optical_data, mask_data)
                
                # Save patches and collect metadata
                patch_metadata = self._save_patches_with_metadata(
                    patches, dataset_name, processed_count, sar_path, optical_path, mask_path
                )
                metadata_list.extend(patch_metadata)
                
                processed_count += 1
                logger.info(f"Processed {processed_count} images from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Error processing image pair: {e}")
                continue
        
        # Generate metadata CSV
        self._generate_metadata_csv(metadata_list, dataset_name)
        
        logger.info(f"Completed processing {dataset_name}: {processed_count} images, {len(metadata_list)} patches")
        return True
    
    def _find_image_pairs(self, dataset_path: str) -> List[Tuple[str, str, str]]:
        """Find matching SAR and optical image pairs in dataset"""
        # This is a simplified implementation
        # In practice, you'd implement dataset-specific logic here
        pairs = []
        
        # Example structure - adapt based on actual dataset organization
        sar_dir = os.path.join(dataset_path, "sar")
        optical_dir = os.path.join(dataset_path, "optical")
        mask_dir = os.path.join(dataset_path, "masks")
        
        if os.path.exists(sar_dir) and os.path.exists(optical_dir):
            sar_files = sorted([f for f in os.listdir(sar_dir) if f.endswith('.tif')])
            optical_files = sorted([f for f in os.listdir(optical_dir) if f.endswith('.tif')])
            
            for sar_file, optical_file in zip(sar_files, optical_files):
                sar_path = os.path.join(sar_dir, sar_file)
                optical_path = os.path.join(optical_dir, optical_file)
                mask_path = os.path.join(mask_dir, sar_file) if os.path.exists(mask_dir) else None
                
                pairs.append((sar_path, optical_path, mask_path))
        
        return pairs
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load segmentation mask"""
        try:
            with rasterio.open(mask_path) as src:
                return src.read(1)  # Read first band
        except Exception as e:
            logger.error(f"Error loading mask from {mask_path}: {e}")
            return None
    
    def _save_patches_with_metadata(self, patches: List[Dict], dataset_name: str, 
                                  image_id: int, sar_path: str, optical_path: str, 
                                  mask_path: Optional[str]) -> List[Dict]:
        """Save processed patches and return metadata"""
        output_dir = os.path.join(self.config["output"]["processed_path"], "patches")
        os.makedirs(output_dir, exist_ok=True)
        
        metadata_list = []
        
        for i, patch in enumerate(patches):
            patch_filename = f"{dataset_name}_{image_id}_{i}.npz"
            patch_path = os.path.join(output_dir, patch_filename)
            
            # Save patch data
            np.savez_compressed(
                patch_path,
                sar=patch['sar'],
                optical=patch['optical'],
                coordinates=patch['coordinates'],
                mask=patch.get('mask', None)
            )
            
            # Create metadata entry
            metadata_entry = {
                'patch_id': patch['patch_id'],
                'dataset': dataset_name,
                'image_id': image_id,
                'patch_index': i,
                'sar_path': sar_path,
                'optical_path': optical_path,
                'mask_path': mask_path if mask_path else '',
                'patch_path': patch_path,
                'coordinates': patch['coordinates'],
                'patch_size': self.config["processing"]["patch_size"],
                'overlap': self.config["processing"]["overlap"],
                'has_mask': mask_path is not None,
                'processing_date': datetime.now().isoformat(),
                'normalization': self.config["processing"]["normalization"],
                'target_resolution': self.config["processing"]["target_resolution"]
            }
            
            metadata_list.append(metadata_entry)
        
        return metadata_list
    
    def _generate_metadata_csv(self, metadata_list: List[Dict], dataset_name: str):
        """Generate metadata CSV file"""
        if not metadata_list:
            logger.warning(f"No metadata to save for {dataset_name}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(metadata_list)
        
        # Save to CSV
        csv_path = os.path.join(
            self.config["output"]["processed_path"], 
            f"{dataset_name}_metadata.csv"
        )
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Generated metadata CSV: {csv_path} with {len(metadata_list)} entries")
        
        # Also append to main metadata file
        main_csv_path = self.config["output"].get("metadata_csv_path", "./data/metadata.csv")
        os.makedirs(os.path.dirname(main_csv_path), exist_ok=True)
        
        if os.path.exists(main_csv_path):
            # Append to existing file
            df.to_csv(main_csv_path, mode='a', header=False, index=False)
        else:
            # Create new file
            df.to_csv(main_csv_path, index=False)
        
        logger.info(f"Updated main metadata CSV: {main_csv_path}")

def main():
    """Main function to run the ETL pipeline"""
    pipeline = FloodETLPipeline()
    
    # Process all datasets (matching actual project implementation)
    datasets = ["sentinel_data", "sen12_flood", "sen1floods11"]
    
    for dataset in datasets:
        success = pipeline.process_dataset(dataset)
        if success:
            logger.info(f"Successfully processed {dataset}")
        else:
            logger.warning(f"Failed to process {dataset} or dataset not found")

if __name__ == "__main__":
    main()
