import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.dataset import Sen1Floods11Dataset
from src.data.transforms import get_val_transforms
from src.models.decoders import FloodNet

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Loading model from {args.checkpoint}...")
    model = FloodNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # Load dataset
    ds = Sen1Floods11Dataset(args.data_dir, split="test", transform=get_val_transforms())
    
    # We want to find interesting samples (e.g., cloudy ones)
    # But since we can't easily query "cloudy" without reading S2, let's just pick random ones
    # or let the user specify indices. For now, random selection.
    indices = np.random.choice(len(ds), args.num_samples, replace=False)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for idx in indices:
        batch = ds[idx]
        s1 = batch['s1'].unsqueeze(0).to(device) # (1, 2, H, W)
        s2 = batch['s2'].unsqueeze(0).to(device) # (1, 13, H, W)
        label = batch['label'].numpy()           # (H, W)
        region_id = batch['id']
        
        with torch.no_grad():
            output, attention = model(s1, s2, return_attention=True)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0] # (H, W)
            attention = attention.cpu().numpy()[0, 0] # (H, W) - First channel, First batch
            
        # Prepare for Plotting
        # S2 (RGB) -> Bands 3, 2, 1 (Red, Green, Blue) -> Indices 2, 1, 0
        s2_img = batch['s2'].cpu().numpy() # (13, H, W)
        rgb = s2_img[[3, 2, 1], :, :] # (3, H, W)
        rgb = np.moveaxis(rgb, 0, -1) # (H, W, 3)
        # Clip to make it look nice (S2 values are 0-1 but often very dark)
        rgb = np.clip(rgb * 3.0, 0, 1) # Brighten
        
        # S1 (SAR) -> VV Channel -> Index 0
        s1_img = batch['s1'].cpu().numpy()
        sar = s1_img[0, :, :] # (H, W)
        
        # Plot
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # 1. Optical (RGB)
        axes[0].imshow(rgb)
        axes[0].set_title(f"Optical (RGB)\n{region_id}")
        axes[0].axis('off')
        
        # 2. SAR (VV)
        axes[1].imshow(sar, cmap='gray')
        axes[1].set_title("SAR (VV)")
        axes[1].axis('off')
        
        # 3. Attention Map
        # 0 (Dark) = Trust SAR, 1 (Bright) = Trust Optical
        im_att = axes[2].imshow(attention, cmap='plasma', vmin=0, vmax=1)
        axes[2].set_title("Attention Map\n(Dark=SAR, Bright=Optical)")
        axes[2].axis('off')
        plt.colorbar(im_att, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 4. Ground Truth
        axes[3].imshow(label, cmap='Blues', interpolation='nearest')
        axes[3].set_title("Ground Truth (Water)")
        axes[3].axis('off')
        
        # 5. Prediction
        axes[4].imshow(pred, cmap='Blues', interpolation='nearest')
        axes[4].set_title("Prediction")
        axes[4].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f"{region_id}_viz.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sen1floods11", help="Path to dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to Gated Fusion model (.pth)")
    parser.add_argument("--save_dir", default="visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize")
    args = parser.parse_args()
    
    visualize(args)
