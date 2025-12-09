import sys
sys.path.append('/home/batman/msml640/FloodVision')

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter

from dataset import get_dataloaders
from models.cloud_aware_fusion import CloudAwareFusionModel

def add_synthetic_clouds(optical, sar, label, cloud_coverage=0.5):
    """Generate and apply synthetic clouds"""
    H, W = label.shape
    noise = np.random.randn(H, W)
    noise = gaussian_filter(noise, sigma=10)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    threshold = np.percentile(noise, (1 - cloud_coverage) * 100)
    cloud_mask = (noise > threshold).astype(np.float32)
    cloud_mask = gaussian_filter(cloud_mask, sigma=3)
    cloud_mask = np.clip(cloud_mask, 0, 1)
    
    C = optical.shape[0]
    cloudy_optical = optical.copy()
    for c in range(C):
        cloudy_optical[c] = optical[c] * (1 - cloud_mask) + 0.8 * cloud_mask
    
    return cloudy_optical, sar, label, cloud_mask

def calculate_iou(pred, target):
    if isinstance(pred, tuple):
        pred = pred[0]
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def load_baseline_model(model_type, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'optical':
        model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=13, classes=1, activation=None)
    elif model_type == 'sar':
        model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=2, classes=1, activation=None)
    elif model_type == 'fusion':
        model = smp.Unet(encoder_name='resnet34', encoder_weights=None,
                        in_channels=15, classes=1, activation=None)
    
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    return model.to(device)

def load_cloud_aware_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CloudAwareFusionModel(encoder_name='resnet34', pretrained=False)
    
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    return model.to(device)

def evaluate_with_clouds(model, model_type, dataloader, device, cloud_coverages=[0.0, 0.3, 0.5, 0.7]):
    model.eval()
    results = {cov: [] for cov in cloud_coverages}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Evaluating {model_type}'):
            optical_np = batch['optical'].cpu().numpy()
            sar_np = batch['sar'].cpu().numpy()
            label_np = batch['label'].cpu().numpy()
            
            for i in range(optical_np.shape[0]):
                for coverage in cloud_coverages:
                    if coverage == 0.0:
                        test_optical = torch.FloatTensor(optical_np[i:i+1]).to(device)
                        test_sar = torch.FloatTensor(sar_np[i:i+1]).to(device)
                        test_label = torch.FloatTensor(label_np[i:i+1]).to(device)
                        cloud_mask = torch.zeros_like(test_label)
                    else:
                        cloudy_opt, _, _, mask = add_synthetic_clouds(
                            optical_np[i], sar_np[i], label_np[i, 0],
                            cloud_coverage=coverage
                        )
                        test_optical = torch.FloatTensor(cloudy_opt[None, :]).to(device)
                        test_sar = torch.FloatTensor(sar_np[i:i+1]).to(device)
                        test_label = torch.FloatTensor(label_np[i:i+1]).to(device)
                        cloud_mask = torch.FloatTensor(mask[None, None, :]).to(device)
                    
                    test_batch = {
                        'optical': test_optical,
                        'sar': test_sar,
                        'label': test_label,
                        'cloud_mask': cloud_mask
                    }
                    
                    # Forward pass based on model type
                    if model_type == 'optical':
                        pred = model(test_batch['optical'])
                    elif model_type == 'sar':
                        pred = model(test_batch['sar'])
                    elif model_type == 'fusion':
                        fused = torch.cat([test_batch['optical'], test_batch['sar']], dim=1)
                        pred = model(fused)
                    else:  # cloud_aware
                        pred, _ = model(test_batch)
                    
                    iou = calculate_iou(pred, test_label).item()
                    results[coverage].append(iou)
    
    return {cov: np.mean(ious) for cov, ious in results.items()}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    _, _, test_loader = get_dataloaders(
        data_dir="/home/batman/scratch.msml640/data/sen1flood11",
        batch_size=8,
        num_workers=4
    )
    
    models = {
        'Optical-Only': ('baseline', 'optical', 
                        Path('~/scratch.msml640/checkpoints/baseline_optical/best_model.pth').expanduser()),
        'SAR-Only': ('baseline', 'sar',
                    Path('~/scratch.msml640/checkpoints/baseline_sar/best_model.pth').expanduser()),
        'Simple Fusion': ('baseline', 'fusion',
                         Path('~/scratch.msml640/checkpoints/baseline_fusion/best_model.pth').expanduser()),
        'Cloud-Aware V1': ('cloud_aware', 'cloud_aware',
                          Path('~/scratch.msml640/checkpoints/cloud_aware_fusion/best_model.pth').expanduser()),
        'Cloud-Aware V2': ('cloud_aware', 'cloud_aware',
                          Path('~/scratch.msml640/checkpoints/cloud_aware_augmented/best_model.pth').expanduser())
    }
    
    all_results = {}
    cloud_coverages = [0.0, 0.3, 0.5, 0.7]
    
    for name, (mclass, mtype, ckpt) in models.items():
        if not ckpt.exists():
            print(f" Skipping {name} - checkpoint not found\n")
            continue
        
        print(f"{'='*70}")
        print(f"Evaluating: {name}")
        print(f"{'='*70}")
        
        if mclass == 'baseline':
            model = load_baseline_model(mtype, ckpt, device)
        else:
            model = load_cloud_aware_model(ckpt, device)
        
        results = evaluate_with_clouds(model, mtype, test_loader, device, cloud_coverages)
        all_results[name] = results
        
        print(f"\nResults:")
        for cov, iou in results.items():
            print(f"  {int(cov*100):3d}% clouds: IoU = {iou:.4f}")
        print()
    
    # Save
    output_dir = Path("~/scratch.msml640/results").expanduser()
    with open(output_dir / 'all_models_cloud_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print table
    print(f"\n{'='*90}")
    print("COMPREHENSIVE CLOUD ROBUSTNESS COMPARISON")
    print(f"{'='*90}")
    print(f"{'Model':<20} {'Clear':<12} {'30% Cloud':<12} {'50% Cloud':<12} {'70% Cloud':<12}")
    print("-" * 90)
    
    for name, results in all_results.items():
        row = f"{name:<20}"
        for cov in cloud_coverages:
            row += f" {results[cov]:<12.4f}"
        print(row)
    
    print(f"{'='*90}")
    print(f"\n✓ Results saved to: {output_dir / 'all_models_cloud_comparison.json'}")

if __name__ == '__main__':
    main()
