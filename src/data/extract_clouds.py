import os
import tacoreader
import rasterio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_clouds(output_dir="data/cloud_bank", num_samples=100):
    """
    Extracts real cloud masks/textures from CloudSEN12 for use in augmentation.
    """
    # 1. Load Dataset
    # We point to the folder where we downloaded the .taco files
    # Note: tacoreader expects the path to the .taco file or folder
    # Assuming files are in data/cloudsen12/
    data_path = "data/cloudsen12/cloudsen12-l1c.0000.part.taco"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        print("Please ensure download_datasets.py cloudsen12 has finished.")
        return

    print("Loading CloudSEN12 dataset via tacoreader...")
    # tacoreader.load can take a list of files if split
    # We'll try loading the folder or specific file list
    try:
        # If we have multiple parts, list them
        files = sorted([
            os.path.join("data/cloudsen12", f) 
            for f in os.listdir("data/cloudsen12") 
            if f.endswith(".taco")
        ])
        dataset = tacoreader.load(files)
    except Exception as e:
        print(f"Failed to load via tacoreader: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting {num_samples} cloudy samples to {output_dir}...")
    
    count = 0
    # Iterate through dataset
    # We don't know exact length easily, so we loop indices
    for idx in tqdm(range(len(dataset))):
        if count >= num_samples:
            break
            
        try:
            # Read metadata/label first to check if it's actually cloudy
            # 0 = L1C image, 1 = Label (Cloud Mask)
            # The label is usually: 0=Clear, 1=Thick Cloud, 2=Thin Cloud, 3=Shadow
            label_handle = dataset.read(idx).read(1)
            
            with rasterio.open(label_handle) as src:
                # Read low res overview first to check cloud cover?
                # Or just read full 512x512
                mask = src.read(1)
                
            # Check cloud percentage (Class 1 and 2)
            cloud_pixels = np.sum((mask == 1) | (mask == 2))
            total_pixels = mask.size
            cloud_cover = cloud_pixels / total_pixels
            
            # We want "interesting" clouds, not full overcast (too easy) or clear (useless)
            if 0.1 < cloud_cover < 0.8:
                # Good candidate! Save it.
                
                # Read Optical Image (L1C)
                img_handle = dataset.read(idx).read(0)
                with rasterio.open(img_handle) as src:
                    # Read RGB bands (usually 4,3,2 for S2)
                    # But just reading all bands is safer for storage
                    image = src.read() # (C, H, W)
                
                # Save to disk
                save_id = f"cloud_{idx}"
                
                # Save Mask
                mask_path = os.path.join(output_dir, f"{save_id}_mask.tif")
                with rasterio.open(
                    mask_path, 'w',
                    driver='GTiff',
                    height=mask.shape[0], width=mask.shape[1],
                    count=1, dtype=mask.dtype
                ) as dst:
                    dst.write(mask, 1)
                    
                # Save Image (Optional, for texture pasting)
                # We save it so we can paste "real cloud pixels", not just white color
                img_path = os.path.join(output_dir, f"{save_id}_img.tif")
                with rasterio.open(
                    img_path, 'w',
                    driver='GTiff',
                    height=image.shape[1], width=image.shape[2],
                    count=image.shape[0], dtype=image.dtype
                ) as dst:
                    dst.write(image)
                
                count += 1
                
        except Exception as e:
            print(f"Skipping index {idx}: {e}")
            continue

    print(f"Done! Saved {count} cloud samples.")

if __name__ == "__main__":
    extract_clouds()

