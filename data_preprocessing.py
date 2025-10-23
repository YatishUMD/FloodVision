"""
Data Preprocessing Utilities for Flood Vision Project
Handles SAR and optical imagery preprocessing, augmentation, and normalization
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy import ndimage
from skimage import exposure, filters
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Handles preprocessing of SAR and optical satellite imagery
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.sar_normalization = config.get('sar_normalization', 'minmax')
        self.optical_normalization = config.get('optical_normalization', 'minmax')
        
    def preprocess_sar(self, sar_data: np.ndarray) -> np.ndarray:
        """
        Preprocess SAR data (Sentinel-1)
        
        Args:
            sar_data: Raw SAR data with shape (bands, height, width)
            
        Returns:
            Preprocessed SAR data
        """
        # Convert to dB scale if not already
        if sar_data.max() > 100:  # Likely in linear scale
            sar_data = 10 * np.log10(np.maximum(sar_data, 1e-10))
        
        # Apply speckle filtering
        sar_data = self._apply_speckle_filter(sar_data)
        
        # Normalize
        if self.sar_normalization == 'minmax':
            sar_data = self._minmax_normalize(sar_data)
        elif self.sar_normalization == 'zscore':
            sar_data = self._zscore_normalize(sar_data)
        elif self.sar_normalization == 'percentile':
            sar_data = self._percentile_normalize(sar_data)
        
        return sar_data
    
    def preprocess_optical(self, optical_data: np.ndarray) -> np.ndarray:
        """
        Preprocess optical data (Sentinel-2)
        
        Args:
            optical_data: Raw optical data with shape (bands, height, width)
            
        Returns:
            Preprocessed optical data
        """
        # Convert to reflectance if needed
        if optical_data.max() > 1.0:
            optical_data = optical_data.astype(np.float32) / 10000.0
        
        # Apply atmospheric correction (simplified)
        optical_data = self._apply_atmospheric_correction(optical_data)
        
        # Normalize
        if self.optical_normalization == 'minmax':
            optical_data = self._minmax_normalize(optical_data)
        elif self.optical_normalization == 'zscore':
            optical_data = self._zscore_normalize(optical_data)
        elif self.optical_normalization == 'histogram':
            optical_data = self._histogram_equalize(optical_data)
        
        return optical_data
    
    def _apply_speckle_filter(self, sar_data: np.ndarray) -> np.ndarray:
        """Apply Lee filter to reduce speckle noise"""
        filtered_data = np.zeros_like(sar_data)
        
        for i in range(sar_data.shape[0]):
            # Convert to linear scale for filtering
            linear_data = np.power(10, sar_data[i] / 10.0)
            
            # Apply Lee filter
            filtered_linear = filters.lee(linear_data, win_size=3)
            
            # Convert back to dB
            filtered_data[i] = 10 * np.log10(np.maximum(filtered_linear, 1e-10))
        
        return filtered_data
    
    def _apply_atmospheric_correction(self, optical_data: np.ndarray) -> np.ndarray:
        """Apply simplified atmospheric correction"""
        # Dark Object Subtraction (DOS)
        corrected_data = np.zeros_like(optical_data)
        
        for i in range(optical_data.shape[0]):
            band_data = optical_data[i]
            
            # Find dark pixels (lowest 1% of values)
            dark_threshold = np.percentile(band_data, 1)
            dark_pixels = band_data <= dark_threshold
            
            if np.any(dark_pixels):
                dark_value = np.mean(band_data[dark_pixels])
                corrected_data[i] = np.maximum(band_data - dark_value, 0)
            else:
                corrected_data[i] = band_data
        
        return corrected_data
    
    def _minmax_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return data
    
    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val > 0:
            return (data - mean_val) / std_val
        else:
            return data - mean_val
    
    def _percentile_normalize(self, data: np.ndarray, 
                            lower_percentile: float = 2.0, 
                            upper_percentile: float = 98.0) -> np.ndarray:
        """Percentile-based normalization"""
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        
        normalized = np.clip(data, lower_bound, upper_bound)
        normalized = (normalized - lower_bound) / (upper_bound - lower_bound)
        
        return normalized
    
    def _histogram_equalize(self, data: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to optical data"""
        equalized_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            # Convert to uint8 for histogram equalization
            uint8_data = (data[i] * 255).astype(np.uint8)
            equalized = cv2.equalizeHist(uint8_data)
            equalized_data[i] = equalized.astype(np.float32) / 255.0
        
        return equalized_data

class DataAugmentation:
    """
    Handles data augmentation for training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_prob = config.get('augmentation_prob', 0.5)
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
        ])
        
        # Define geometric transformations for SAR data
        self.geometric_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.3
            ),
        ])
    
    def augment_data(self, sar_data: np.ndarray, optical_data: np.ndarray, 
                    mask_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Apply data augmentation
        
        Args:
            sar_data: SAR data with shape (bands, height, width)
            optical_data: Optical data with shape (bands, height, width)
            mask_data: Optional mask data with shape (height, width)
            
        Returns:
            Augmented data tuples
        """
        if np.random.random() > self.augmentation_prob:
            return sar_data, optical_data, mask_data
        
        # Convert to format expected by albumentations
        # Transpose to (height, width, channels)
        sar_transposed = np.transpose(sar_data, (1, 2, 0))
        optical_transposed = np.transpose(optical_data, (1, 2, 0))
        
        # Apply geometric transformations
        if mask_data is not None:
            augmented = self.geometric_augmentation(
                image=optical_transposed,
                mask=mask_data
            )
            optical_transposed = augmented['image']
            mask_data = augmented['mask']
            
            # Apply same transformation to SAR
            sar_transposed = self.geometric_augmentation(
                image=sar_transposed
            )['image']
        else:
            augmented = self.geometric_augmentation(
                image=optical_transposed
            )
            optical_transposed = augmented['image']
            
            sar_transposed = self.geometric_augmentation(
                image=sar_transposed
            )['image']
        
        # Apply photometric transformations to optical data only
        optical_transposed = self.augmentation_pipeline(
            image=optical_transposed
        )['image']
        
        # Convert back to (channels, height, width)
        sar_data = np.transpose(sar_transposed, (2, 0, 1))
        optical_data = np.transpose(optical_transposed, (2, 0, 1))
        
        return sar_data, optical_data, mask_data

class DataAlignment:
    """
    Handles alignment of SAR and optical imagery
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.target_resolution = config.get('target_resolution', 10)
        self.interpolation_method = config.get('interpolation_method', 'bilinear')
    
    def align_resolution(self, sar_data: np.ndarray, optical_data: np.ndarray,
                        sar_resolution: float, optical_resolution: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align SAR and optical data to same resolution
        
        Args:
            sar_data: SAR data
            optical_data: Optical data
            sar_resolution: SAR resolution in meters
            optical_resolution: Optical resolution in meters
            
        Returns:
            Aligned data tuples
        """
        # Determine which dataset needs resampling
        if abs(sar_resolution) != self.target_resolution:
            sar_data = self._resample_to_target_resolution(sar_data, sar_resolution)
        
        if abs(optical_resolution) != self.target_resolution:
            optical_data = self._resample_to_target_resolution(optical_data, optical_resolution)
        
        return sar_data, optical_data
    
    def _resample_to_target_resolution(self, data: np.ndarray, current_resolution: float) -> np.ndarray:
        """Resample data to target resolution"""
        scale_factor = current_resolution / self.target_resolution
        
        if abs(scale_factor - 1.0) < 0.01:  # Already at target resolution
            return data
        
        new_height = int(data.shape[1] * scale_factor)
        new_width = int(data.shape[2] * scale_factor)
        
        resampled_data = np.zeros((data.shape[0], new_height, new_width), dtype=data.dtype)
        
        for i in range(data.shape[0]):
            if self.interpolation_method == 'bilinear':
                resampled_data[i] = cv2.resize(
                    data[i], 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_LINEAR
                )
            elif self.interpolation_method == 'nearest':
                resampled_data[i] = cv2.resize(
                    data[i], 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_NEAREST
                )
            elif self.interpolation_method == 'cubic':
                resampled_data[i] = cv2.resize(
                    data[i], 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_CUBIC
                )
        
        return resampled_data
    
    def align_spatial_extent(self, sar_data: np.ndarray, optical_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align spatial extents of SAR and optical data"""
        # Find minimum dimensions
        min_height = min(sar_data.shape[1], optical_data.shape[1])
        min_width = min(sar_data.shape[2], optical_data.shape[2])
        
        # Crop both datasets to minimum dimensions
        sar_cropped = sar_data[:, :min_height, :min_width]
        optical_cropped = optical_data[:, :min_height, :min_width]
        
        return sar_cropped, optical_cropped
    
    def create_aligned_patches(self, sar_data: np.ndarray, optical_data: np.ndarray,
                             patch_size: int, overlap: float = 0.0) -> List[Dict]:
        """
        Create aligned patches from full images
        
        Args:
            sar_data: SAR data
            optical_data: Optical data
            patch_size: Size of patches
            overlap: Overlap between patches (0.0 to 0.5)
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        step_size = int(patch_size * (1 - overlap))
        
        height, width = sar_data.shape[1], sar_data.shape[2]
        
        for i in range(0, height - patch_size + 1, step_size):
            for j in range(0, width - patch_size + 1, step_size):
                patch_data = {
                    'sar': sar_data[:, i:i+patch_size, j:j+patch_size],
                    'optical': optical_data[:, i:i+patch_size, j:j+patch_size],
                    'coordinates': (i, j),
                    'patch_id': f"{i}_{j}"
                }
                patches.append(patch_data)
        
        return patches

class DataNormalization:
    """
    Handles various normalization strategies
    """
    
    @staticmethod
    def normalize_by_dataset_statistics(data: np.ndarray, 
                                       mean: np.ndarray, 
                                       std: np.ndarray) -> np.ndarray:
        """Normalize using dataset statistics"""
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            if std[i] > 0:
                normalized_data[i] = (data[i] - mean[i]) / std[i]
            else:
                normalized_data[i] = data[i] - mean[i]
        
        return normalized_data
    
    @staticmethod
    def denormalize_by_dataset_statistics(normalized_data: np.ndarray,
                                        mean: np.ndarray,
                                        std: np.ndarray) -> np.ndarray:
        """Denormalize using dataset statistics"""
        denormalized_data = np.zeros_like(normalized_data)
        
        for i in range(normalized_data.shape[0]):
            denormalized_data[i] = normalized_data[i] * std[i] + mean[i]
        
        return denormalized_data
    
    @staticmethod
    def compute_dataset_statistics(data_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std across a list of data arrays"""
        all_data = np.concatenate([d.reshape(d.shape[0], -1) for d in data_list], axis=1)
        
        mean = np.mean(all_data, axis=1)
        std = np.std(all_data, axis=1)
        
        return mean, std

def create_preprocessing_pipeline(config: Dict) -> Dict:
    """
    Create a complete preprocessing pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing preprocessing components
    """
    pipeline = {
        'preprocessor': ImagePreprocessor(config),
        'augmentation': DataAugmentation(config),
        'alignment': DataAlignment(config),
        'normalization': DataNormalization()
    }
    
    return pipeline
