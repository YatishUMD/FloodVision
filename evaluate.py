import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
from torch.utils.data import DataLoader

from src.data.dataset import Sen1Floods11Dataset
from src.data.transforms import get_val_transforms
from src.models.decoders import FloodNet

def calculate_iou(pred, label):
    # Flatten
    pred = pred.view(-1)
    label = label.view(-1)
    
    # Intersection & Union for Class 1 (Water)
    intersection = ((pred == 1) & (label == 1)).sum().item()
    union = ((pred == 1) | (label == 1)).sum().item()
    
    return intersection, union

def estimate_cloud_cover(s2_path):
    """
    Heuristic to estimate cloud cover if we don't have a perfect mask.
    Simple method: Percentage of bright white pixels in S2.
    """
    try:
        with rasterio.open(s2_path) as src:
            img = src.read() # (C, H, W)
            # S2 values are 0-10000. Clouds are bright (> 2000-3000 in all bands usually)
            # Simple threshold on Blue/Green/Red (Bands 2,3,4 -> Indices 1,2,3 in 13-band)
            # Let's just take mean across channels for simplicity
            mean_intensity = np.mean(img, axis=0)
            cloud_pixels = np.sum(mean_intensity > 2500) # Threshold for "Bright"
            return cloud_pixels / img.size
    except:
        return 0.0

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Loading model from {args.checkpoint}...")
    model = FloodNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    test_ds = Sen1Floods11Dataset(args.data_dir, split="test", transform=get_val_transforms())
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    print(f"Evaluating on {len(test_ds)} samples...")
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            s1 = batch['s1'].to(device)
            s2 = batch['s2'].to(device)
            label = batch['label'].to(device)
            region_id = batch['id'][0] # String ID
            
            # Predict
            outputs = model(s1, s2)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate IoU
            inter, union = calculate_iou(preds, label)
            iou = inter / (union + 1e-6)
            
            # Estimate Cloud Cover for Stratification
            # We need the path to S2 image to estimate brightness
            # Reconstruct path from dataset logic
            # This is a bit hacky, better if dataset returned cloud info
            # But let's look it up
            s2_filename = f"{region_id}_S2Hand.tif"
            s2_path = os.path.join(args.data_dir, "v1.1/data/flood_events/HandLabeled/S2Hand", s2_filename)
            cloud_pct = estimate_cloud_cover(s2_path)
            
            results.append({
                "id": region_id,
                "iou": iou,
                "cloud_pct": cloud_pct,
                "intersection": inter,
                "union": union
            })
            
    df = pd.DataFrame(results)
    
    # Stratified Reporting
    # Clear: < 10% Cloud
    # Cloudy: > 20% Cloud (adjust thresholds as needed)
    
    clear_subset = df[df["cloud_pct"] < 0.1]
    cloudy_subset = df[df["cloud_pct"] >= 0.2]
    
    overall_iou = df["intersection"].sum() / df["union"].sum()
    clear_iou = clear_subset["intersection"].sum() / clear_subset["union"].sum()
    cloudy_iou = cloudy_subset["intersection"].sum() / cloudy_subset["union"].sum()
    
    print("\n--- Evaluation Results ---")
    print(f"Overall mIoU: {overall_iou:.4f}")
    print(f"Clear Images (n={len(clear_subset)}) mIoU:  {clear_iou:.4f}")
    print(f"Cloudy Images (n={len(cloudy_subset)}) mIoU: {cloudy_iou:.4f}")
    
    # Save per-image results
    df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sen1floods11", help="Path to dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    args = parser.parse_args()
    
    evaluate(args)

