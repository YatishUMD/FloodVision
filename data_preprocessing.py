import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import cv2
from patchify import patchify
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
SAR_PATH = "data/raw/SAR/"          # folder with Sentinel-1 VV/VH GeoTIFFs
OPTICAL_PATH = "data/raw/OPTICAL/"  # folder with Sentinel-2 RGB/NIR GeoTIFFs
MASK_PATH = "data/raw/MASKS/"       # folder with flood masks
OUTPUT_DIR = "data/processed/"
PATCH_SIZE = 256
STEP = 128
REGION_NAME = "West_Africa"

os.makedirs(f"{OUTPUT_DIR}/patches/sar", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/patches/optical", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/patches/masks", exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def normalize_band(band):
    """Normalize band using z-score"""
    return (band - np.mean(band)) / (np.std(band) + 1e-8)

def read_raster(path):
    with rasterio.open(path) as src:
        img = src.read(out_dtype=np.float32)
        profile = src.profile
    return img, profile

def resample_to_match(src_path, ref_profile):
    """Resample and reproject raster to match reference profile"""
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(src.count,
                       int(ref_profile["height"]),
                       int(ref_profile["width"])),
            resampling=Resampling.bilinear
        )
        transform = ref_profile["transform"]
        profile = src.profile
        profile.update({
            "height": ref_profile["height"],
            "width": ref_profile["width"],
            "transform": transform
        })
    return data, profile

def align_and_stack(sar_vv_path, sar_vh_path, optical_paths, mask_path):
    # --- Read SAR ---
    sar_vv, ref_profile = read_raster(sar_vv_path)
    sar_vh, _ = resample_to_match(sar_vh_path, ref_profile)
    sar_stack = np.stack([normalize_band(sar_vv[0]), normalize_band(sar_vh[0])], axis=-1)

    # --- Read OPTICAL (RGB or RGB+NIR) ---
    optical_bands = []
    for path in optical_paths:
        band, _ = resample_to_match(path, ref_profile)
        optical_bands.append(normalize_band(band[0]))
    optical_stack = np.stack(optical_bands, axis=-1)

    # --- Read MASK ---
    mask, _ = resample_to_match(mask_path, ref_profile)
    mask = (mask[0] > 0.5).astype(np.uint8)

    return sar_stack, optical_stack, mask

def create_patches(image, mask, modality, out_dir, patch_size=256, step=128, prefix=""):
    patches = patchify(image, (patch_size, patch_size, image.shape[2]), step=step)
    patches_mask = patchify(mask, (patch_size, patch_size), step=step)

    count = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            img_patch = patches[i, j, 0]
            mask_patch = patches_mask[i, j, 0]
            if mask_patch.sum() == 0:  # skip empty flood masks
                continue
            cv2.imwrite(f"{out_dir}/optical/{prefix}_{count}.png",
                        cv2.cvtColor((img_patch[:, :, :3]*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            np.save(f"{out_dir}/sar/{prefix}_{count}.npy", img_patch[:, :, -2:])  # store SAR separately
            cv2.imwrite(f"{out_dir}/masks/{prefix}_{count}.png", (mask_patch*255).astype(np.uint8))
            count += 1
    return count

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def preprocess_region(region_name=REGION_NAME):
    metadata_records = []
    sar_files = sorted([f for f in os.listdir(SAR_PATH) if "VV" in f])

    print(f"Processing region: {region_name}")

    for sar_vv_file in tqdm(sar_files):
        prefix = sar_vv_file.replace("_VV.tif", "")
        sar_vh_file = f"{prefix}_VH.tif"
        mask_file = f"{prefix}_mask.tif"

        optical_files = [
            f"{OPTICAL_PATH}{prefix}_B4.tif",  # Red
            f"{OPTICAL_PATH}{prefix}_B3.tif",  # Green
            f"{OPTICAL_PATH}{prefix}_B2.tif"   # Blue
        ]

        sar_vv_path = os.path.join(SAR_PATH, sar_vv_file)
        sar_vh_path = os.path.join(SAR_PATH, sar_vh_file)
        mask_path = os.path.join(MASK_PATH, mask_file)

        try:
            sar_stack, optical_stack, mask = align_and_stack(
                sar_vv_path, sar_vh_path, optical_files, mask_path
            )
        except Exception as e:
            print(f"Skipping {prefix}: {e}")
            continue

        # Combine into single 5-channel stack
        combined = np.concatenate([sar_stack, optical_stack], axis=-1)
        n_patches = create_patches(combined, mask, "combined", OUTPUT_DIR, PATCH_SIZE, STEP, prefix)

        metadata_records.append({
            "tile_id": prefix,
            "region": region_name,
            "num_patches": n_patches
        })

    # Save metadata
    df = pd.DataFrame(metadata_records)
    df.to_csv(f"{OUTPUT_DIR}/metadata_{region_name}.csv", index=False)
    print(f" Preprocessing complete for {region_name}")

if __name__ == "__main__":
    preprocess_region()
