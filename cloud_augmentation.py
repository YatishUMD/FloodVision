import numpy as np
import torch
from scipy.ndimage import gaussian_filter

def generate_cloud_mask(shape, coverage=0.5, complexity=3):
    """
    Generate realistic cloud mask using multiple octaves of noise
    
    Args:
        shape: (H, W)
        coverage: Target cloud coverage (0-1)
        complexity: Number of noise octaves (higher = more realistic)
    """
    H, W = shape
    cloud_mask = np.zeros((H, W))
    
    # Multi-scale Perlin-like noise
    for octave in range(complexity):
        scale = 2 ** octave
        noise = np.random.randn(H // scale + 1, W // scale + 1)
        
        # Upsample to full size
        from scipy.ndimage import zoom
        noise_full = zoom(noise, (scale, scale), order=1)[:H, :W]
        
        # Add to cloud mask with decreasing weight
        weight = 1.0 / (2 ** octave)
        cloud_mask += weight * noise_full
    
    # Normalize
    cloud_mask = (cloud_mask - cloud_mask.min()) / (cloud_mask.max() - cloud_mask.min())
    
    # Apply gaussian smoothing for realistic edges
    cloud_mask = gaussian_filter(cloud_mask, sigma=10)
    
    # Threshold to achieve target coverage
    threshold = np.percentile(cloud_mask, (1 - coverage) * 100)
    cloud_mask = (cloud_mask > threshold).astype(np.float32)
    
    # Smooth edges again
    cloud_mask = gaussian_filter(cloud_mask, sigma=3)
    cloud_mask = np.clip(cloud_mask, 0, 1)
    
    return cloud_mask


def apply_clouds_to_optical(optical, cloud_mask, cloud_brightness=0.8):
    """
    Apply cloud mask to optical imagery
    
    Args:
        optical: (C, H, W) numpy array
        cloud_mask: (H, W) numpy array in [0, 1]
        cloud_brightness: Brightness of clouds (0.7-1.0)
    """
    C, H, W = optical.shape
    cloudy_optical = optical.copy()
    
    # Clouds are bright/white across all bands
    for c in range(C):
        # Original * (1 - cloud) + bright_white * cloud
        cloudy_optical[c] = optical[c] * (1 - cloud_mask) + cloud_brightness * cloud_mask
    
    return cloudy_optical


def augment_with_clouds(optical, sar, label, cloud_mask_original, 
                        cloud_probability=0.5, coverage_range=(0.2, 0.7)):
    """
    Augment a sample with synthetic clouds during training
    
    Args:
        optical: (C, H, W) optical data
        sar: (C, H, W) SAR data (unchanged)
        label: (H, W) ground truth
        cloud_mask_original: (H, W) original cloud mask
        cloud_probability: Probability of adding clouds (0.5 = 50% of time)
        coverage_range: (min, max) cloud coverage
    
    Returns:
        optical (possibly clouded), sar, label, cloud_mask (updated)
    """
    if np.random.random() < cloud_probability:
        # Add synthetic clouds
        H, W = label.shape
        coverage = np.random.uniform(*coverage_range)
        
        # Generate cloud mask
        synthetic_cloud = generate_cloud_mask((H, W), coverage=coverage)
        
        # Apply to optical
        cloudy_optical = apply_clouds_to_optical(optical, synthetic_cloud)
        
        # Update cloud mask (combine with original)
        updated_cloud_mask = np.maximum(cloud_mask_original, synthetic_cloud)
        
        return cloudy_optical, sar, label, updated_cloud_mask
    else:
        # Return original (no clouds)
        return optical, sar, label, cloud_mask_original
