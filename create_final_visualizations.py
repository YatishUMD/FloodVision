import sys
sys.path.append('/home/batman/msml640/FloodVision')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import segmentation_models_pytorch as smp
from scipy.ndimage import gaussian_filter

from dataset import get_dataloaders
from models.cloud_aware_fusion import CloudAwareFusionModel

# Cloud generation function
def add_synthetic_clouds(optical, sar, label, cloud_coverage=0.5):
    """Generate and apply synthetic clouds"""
    H, W = label.shape
    
    # Generate cloud mask
    noise = np.random.randn(H, W)
    noise = gaussian_filter(noise, sigma=10)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    threshold = np.percentile(noise, (1 - cloud_coverage) * 100)
    cloud_mask = (noise > threshold).astype(np.float32)
    cloud_mask = gaussian_filter(cloud_mask, sigma=3)
    cloud_mask = np.clip(cloud_mask, 0, 1)
    
    # Apply to optical
    C = optical.shape[0]
    cloudy_optical = optical.copy()
    for c in range(C):
        cloudy_optical[c] = optical[c] * (1 - cloud_mask) + 0.8 * cloud_mask
    
    return cloudy_optical, sar, label, cloud_mask


def load_baseline_model(model_type, checkpoint_path, device):
    """Load baseline models"""
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
    """Load cloud-aware model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CloudAwareFusionModel(encoder_name='resnet34', pretrained=False)
    
    state_dict = checkpoint['model_state_dict']
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    
    return model.to(device)


def visualize_predictions(output_dir):
    """Create side-by-side prediction comparison"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(
        data_dir="/home/batman/scratch.msml640/data/sen1flood11",
        batch_size=1,
        num_workers=2
    )
    
    # Load models
    print("Loading models...")
    models = {}
    model_configs = {
        'Optical': ('baseline', 'optical', 
                   Path('~/scratch.msml640/checkpoints/baseline_optical/best_model.pth').expanduser()),
        'SAR': ('baseline', 'sar',
               Path('~/scratch.msml640/checkpoints/baseline_sar/best_model.pth').expanduser()),
        'Fusion': ('baseline', 'fusion',
                  Path('~/scratch.msml640/checkpoints/baseline_fusion/best_model.pth').expanduser()),
        'Cloud-Aware V2': ('cloud_aware', None,
                          Path('~/scratch.msml640/checkpoints/cloud_aware_augmented/best_model.pth').expanduser())
    }
    
    for name, (mclass, mtype, ckpt) in model_configs.items():
        if ckpt.exists():
            if mclass == 'baseline':
                models[name] = (load_baseline_model(mtype, ckpt, device), mtype)
            else:
                models[name] = (load_cloud_aware_model(ckpt, device), 'cloud_aware')
            print(f"  ✓ Loaded {name}")
        else:
            print(f"  ⚠ Skipping {name} - checkpoint not found")
    
    if not models:
        print("No models loaded!")
        return
    
    # Create figure - 6 samples × 7 columns
    num_samples = 6
    fig, axes = plt.subplots(num_samples, 7, figsize=(21, 3*num_samples))
    
    print(f"\nGenerating visualizations for {num_samples} samples...")
    
    for row, batch in enumerate(test_loader):
        if row >= num_samples:
            break
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optical_np = batch['optical'][0].cpu().numpy()
        sar_np = batch['sar'][0].cpu().numpy()
        label_np = batch['label'][0, 0].cpu().numpy()
        
        # Column 0: Optical RGB
        rgb = optical_np[[3,2,1], :, :]  # R, G, B bands
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb * 3, 0, 1)  # Brightness adjustment
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title('Optical RGB' if row == 0 else '', fontsize=10)
        axes[row, 0].axis('off')
        
        # Column 1: SAR VH
        sar_vh = sar_np[1, :, :]
        axes[row, 1].imshow(sar_vh, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('SAR (VH)' if row == 0 else '', fontsize=10)
        axes[row, 1].axis('off')
        
        # Column 2: Ground Truth
        axes[row, 2].imshow(label_np, cmap='Blues', vmin=0, vmax=1)
        axes[row, 2].set_title('Ground Truth' if row == 0 else '', fontsize=10)
        axes[row, 2].axis('off')
        
        # Columns 3-6: Model predictions
        with torch.no_grad():
            for col, (model_name, (model, mtype)) in enumerate(models.items(), start=3):
                model.eval()
                
                # Forward pass
                if mtype == 'optical':
                    pred = model(batch['optical'])
                elif mtype == 'sar':
                    pred = model(batch['sar'])
                elif mtype == 'fusion':
                    fused = torch.cat([batch['optical'], batch['sar']], dim=1)
                    pred = model(fused)
                else:  # cloud_aware
                    pred, gate = model(batch)
                
                pred_np = torch.sigmoid(pred[0, 0]).cpu().numpy()
                axes[row, col].imshow(pred_np, cmap='Blues', vmin=0, vmax=1)
                axes[row, col].set_title(model_name if row == 0 else '', fontsize=10)
                axes[row, col].axis('off')
    
    plt.suptitle('Flood Detection: Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'predictions_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def visualize_cloud_robustness(output_dir):
    """Show model performance under varying cloud coverage"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nGenerating cloud robustness visualization...")
    
    # Load data
    _, _, test_loader = get_dataloaders(
        data_dir="/home/batman/scratch.msml640/data/sen1flood11",
        batch_size=1,
        num_workers=2
    )
    
    # Load V2 model
    model_path = Path('~/scratch.msml640/checkpoints/cloud_aware_augmented/best_model.pth').expanduser()
    if not model_path.exists():
        print(f"⚠ V2 model not found at {model_path}")
        return
    
    model = load_cloud_aware_model(model_path, device)
    model.eval()
    
    # Get one sample
    batch = next(iter(test_loader))
    optical_np = batch['optical'][0].cpu().numpy()
    sar_np = batch['sar'][0].cpu().numpy()
    label_np = batch['label'][0, 0].cpu().numpy()
    
    # Create figure: 3 rows × 5 columns
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    coverages = [0.0, 0.3, 0.5, 0.7]
    
    for col, coverage in enumerate(coverages):
        # Generate cloudy version
        if coverage == 0.0:
            cloudy_opt = optical_np
            mask = np.zeros_like(label_np)
        else:
            cloudy_opt, _, _, mask = add_synthetic_clouds(
                optical_np, sar_np, label_np, cloud_coverage=coverage
            )
        
        # Prepare tensors
        test_batch = {
            'optical': torch.FloatTensor(cloudy_opt[None, :]).to(device),
            'sar': torch.FloatTensor(sar_np[None, :]).to(device),
            'label': torch.FloatTensor(label_np[None, None, :]).to(device),
            'cloud_mask': torch.FloatTensor(mask[None, None, :]).to(device)
        }
        
        # Predict
        with torch.no_grad():
            pred, gate = model(test_batch)
            pred_np = torch.sigmoid(pred[0, 0]).cpu().numpy()
            gate_val = gate.item() if isinstance(gate, torch.Tensor) else gate        
        # Row 0: Cloudy optical RGB
        rgb = cloudy_opt[[3,2,1], :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb * 3, 0, 1)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f'{int(coverage*100)}% Cloud Coverage', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
        
        # Row 1: Prediction
        axes[1, col].imshow(pred_np, cmap='Blues', vmin=0, vmax=1)
        axes[1, col].set_title(f'Prediction\n(Gate α={gate_val:.2f})', fontsize=10)
        axes[1, col].axis('off')
        
        # Row 2: Cloud mask
        axes[2, col].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[2, col].set_title(f'Cloud Mask\n({int(mask.mean()*100)}% coverage)', fontsize=10)
        axes[2, col].axis('off')
    
    # Last column: SAR and Ground Truth
    sar_vh = sar_np[1, :, :]
    axes[0, 4].imshow(sar_vh, cmap='gray', vmin=0, vmax=1)
    axes[0, 4].set_title('SAR (VH)\n(Always clear)', fontsize=12, fontweight='bold')
    axes[0, 4].axis('off')
    
    axes[1, 4].imshow(label_np, cmap='Blues', vmin=0, vmax=1)
    axes[1, 4].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1, 4].axis('off')
    
    # Gate explanation
    axes[2, 4].text(0.5, 0.5, 
                   'Gate Weight (α):\n\nα = 0.0 → Trust Optical\nα = 0.5 → Balance\nα = 1.0 → Trust SAR',
                   ha='center', va='center', fontsize=11,
                   transform=axes[2, 4].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[2, 4].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Optical\nImagery', va='center', rotation='vertical', 
            fontsize=12, fontweight='bold')
    fig.text(0.02, 0.50, 'Flood\nPrediction', va='center', rotation='vertical',
            fontsize=12, fontweight='bold')
    fig.text(0.02, 0.25, 'Cloud\nMask', va='center', rotation='vertical',
            fontsize=12, fontweight='bold')
    
    plt.suptitle('Cloud-Aware Fusion: Adaptive Behavior Under Cloud Coverage', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    
    output_path = output_dir / 'cloud_robustness_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_performance_curves(output_dir):
    """Plot IoU vs cloud coverage for all models"""
    print("\nGenerating performance curves...")
    
    # Load results
    results_file = output_dir / 'all_models_cloud_comparison.json'
    if not results_file.exists():
        print(f"⚠ Results file not found: {results_file}")
        print("  Run evaluate_all_models_with_clouds.py first")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    coverages = [0, 30, 50, 70]
    coverage_values = [0.0, 0.3, 0.5, 0.7]
    
    # Color scheme
    colors = {
        'Optical-Only': '#1f77b4',
        'SAR-Only': '#ff7f0e',
        'Simple Fusion': '#2ca02c',
        'Cloud-Aware V1': '#d62728',
        'Cloud-Aware V2': '#9467bd'
    }
    
    # Plot 1: IoU vs Cloud Coverage
    for model_name in results.keys():
        ious = [results[model_name][str(c)] for c in coverage_values]
        color = colors.get(model_name, 'gray')
        linewidth = 3 if 'V2' in model_name else 2
        marker = 'o' if 'V2' in model_name else 's'
        axes[0].plot(coverages, ious, marker=marker, linewidth=linewidth, 
                    markersize=8, label=model_name, color=color)
    
    axes[0].set_xlabel('Cloud Coverage (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Robustness to Cloud Coverage', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, 0.8])
    axes[0].set_xticks(coverages)
    
    # Plot 2: Performance Retention
    for model_name in results.keys():
        clear_iou = results[model_name]['0.0']
        retentions = [(results[model_name][str(c)] / clear_iou * 100) for c in coverage_values]
        color = colors.get(model_name, 'gray')
        linewidth = 3 if 'V2' in model_name else 2
        marker = 'o' if 'V2' in model_name else 's'
        axes[1].plot(coverages, retentions, marker=marker, linewidth=linewidth,
                    markersize=8, label=model_name, color=color)
    
    axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('Cloud Coverage (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Performance Retention (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Performance Retention Relative to Clear Sky', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, 120])
    axes[1].set_xticks(coverages)
    
    plt.tight_layout()
    output_path = output_dir / 'performance_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_metrics_comparison(output_dir):
    """Bar chart comparing all metrics"""
    print("\nGenerating metrics comparison...")
    
    # Load clear-sky results
    results_file = output_dir / 'final_results.json'
    if not results_file.exists():
        print(f"⚠ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    models = list(results.keys())
    iou_scores = [results[m]['all']['iou'] for m in models]
    f1_scores = [results[m]['all']['f1'] for m in models]
    precision_scores = [results[m]['all']['precision'] for m in models]
    recall_scores = [results[m]['all']['recall'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Plot 1: IoU and F1
    bars1 = axes[0].bar(x - width/2, iou_scores, width, label='IoU', 
                       alpha=0.8, color='steelblue')
    bars2 = axes[0].bar(x + width/2, f1_scores, width, label='F1-Score',
                       alpha=0.8, color='coral')
    
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Overall Performance (Clear Sky)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Add value labels
    for i, (iou, f1) in enumerate(zip(iou_scores, f1_scores)):
        axes[0].text(i - width/2, iou + 0.02, f'{iou:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, f1 + 0.02, f'{f1:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Precision and Recall
    bars3 = axes[1].bar(x - width/2, precision_scores, width, label='Precision',
                       alpha=0.8, color='green')
    bars4 = axes[1].bar(x + width/2, recall_scores, width, label='Recall',
                       alpha=0.8, color='purple')
    
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path("~/scratch.msml640/results").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Creating Final Visualizations for FloodVision")
    print("="*70)
    
    # 1. Prediction comparison
    print("\n[1/4] Generating prediction comparison...")
    visualize_predictions(output_dir)
    
    # 2. Cloud robustness demo
    print("\n[2/4] Generating cloud robustness demonstration...")
    visualize_cloud_robustness(output_dir)
    
    # 3. Performance curves
    print("\n[3/4] Generating performance curves...")
    plot_performance_curves(output_dir)
    
    # 4. Metrics comparison
    print("\n[4/4] Generating metrics comparison...")
    plot_metrics_comparison(output_dir)
    
    print("\n" + "="*70)
    print("✓ All visualizations complete!")
    print("="*70)
    print(f"\nGenerated files in {output_dir}/:")
    print("  - predictions_comparison.png")
    print("  - cloud_robustness_demo.png")
    print("  - performance_curves.png")
    print("  - metrics_comparison.png")
    print("\nThese figures are ready for your final report!")


if __name__ == '__main__':
    main()
