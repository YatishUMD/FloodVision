# FloodVision: Cloud-Aware SAR-Optical Fusion for All-Weather Flood Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Course:** MSML640 - Computer Vision  
**Authors:** Yatish Sikka, Ritvik Chaturvedi  
**Institution:** University of Maryland, College Park  
**Date:** December 2025

---

## 📋 Overview

FloodVision is a deep learning system for all-weather flood detection that intelligently combines Sentinel-1 SAR and Sentinel-2 optical satellite imagery. Our cloud-aware gating mechanism learns to adaptively weight each modality based on cloud coverage, enabling reliable flood mapping even during storms when traditional optical systems fail.

### Key Innovation
We discovered that having a gating architecture alone isn't sufficient - the model must be **trained with synthetic cloud augmentation** to learn proper adaptation behavior. This insight led to a 26x performance improvement under heavy cloud conditions.

### Architecture Overview
```
┌─────────────────────────────────────────┐
│  Optical Branch (Sentinel-2, 13 bands)  │
│  └─> UNet → Flood Prediction A         │
├─────────────────────────────────────────┤
│  SAR Branch (Sentinel-1, 2 bands)       │
│  └─> UNet → Flood Prediction B         │
├─────────────────────────────────────────┤
│  Cloud Detection → Gate Network → α    │
└─────────────────────────────────────────┘
           ↓
Final = (1-α) × A + α × B
```

---

## 📁 Project Structure
```
FloodVision/
├── README.md                          
├── requirements.txt                   # Python dependencies
│
├── models/
│   ├── baselines.py                  # Optical, SAR, Simple Fusion
│   └── cloud_aware_fusion.py         # Cloud-aware gated fusion
│
├── dataset.py                    # SEN1Floods11 dataloader
├── dataset_with_clouds.py        # Dataloader with cloud augmentation
└── cloud_augmentation.py         # Synthetic cloud generation
│
├── train.py                           # Training script for baselines
├── train_cloud_aware_2.py           # Training with cloud augmentation
├── evaluate_all_with_clouds.py
├── create_final_visualizations.py    # Generate figures
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 64GB RAM recommended
- ~100GB disk space for dataset

### Setup Environment

#### Option 1: Conda (Recommended)
```bash
# Clone repository
git clone https://github.com/YatishUMD/FloodVision.git
cd FloodVision

# Create conda environment
conda create -n floodvision python=3.10
conda activate floodvision

# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install geospatial and other dependencies
conda install -c conda-forge rasterio gdal opencv matplotlib pandas scikit-learn scipy
pip install segmentation-models-pytorch albumentations tensorboard tqdm
```

#### Option 2: Using requirements.txt
```bash
pip install -r requirements.txt
```

### Environment Variables (HPC/SLURM)
```bash
# Add to your .bashrc or run before training
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 📊 Dataset Setup

### Download SEN1Floods11
```bash
# Create data directory
mkdir -p data/sen1flood11
cd data/sen1flood11

# Download dataset
# Visit: https://github.com/cloudtostreet/Sen1Floods11
# Or use direct link if available
wget https://storage.googleapis.com/sen1floods11/v1.1/data.zip
unzip data.zip

# Verify structure
ls data/flood_events/HandLabeled/
# Expected: S1Hand/ S2Hand/ LabelHand/
```

### Expected Directory Structure
```
data/sen1flood11/
└── data/
    └── flood_events/
        └── HandLabeled/
            ├── S1Hand/          # Sentinel-1 SAR (446 .tif files)
            ├── S2Hand/          # Sentinel-2 Optical (446 .tif files)
            └── LabelHand/       # Ground truth labels (446 .tif files)
```

**Dataset Info:**
- Total samples: 446 (512×512 pixels each)
- Train/Val/Test split: 312 / 67 / 67
- Size: ~33 GB
- Source: [Sen1Floods11 GitHub](https://github.com/cloudtostreet/Sen1Floods11)

---

## 🚀 Quick Start

### Evaluate Pre-trained Models
```bash
# Activate environment
conda activate floodvision
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Comprehensive evaluation (all models, all cloud levels)
python evaluate_all_with_clouds.py

# Generate visualizations
python create_final_visualizations.py
```

### Train from Scratch

#### 1. Train Baseline Models
```bash
# Optical-only
python train.py --model_type optical --output_dir checkpoints/baseline_optical --epochs 50

# SAR-only
python train.py --model_type sar --output_dir checkpoints/baseline_sar --epochs 50

# Simple fusion
python train.py --model_type fusion --output_dir checkpoints/baseline_fusion --epochs 50
```

#### 2. Train Cloud-Aware 2 (Recommended)
```bash
python train_cloud_aware_2.py \
    --output_dir checkpoints/cloud_aware_v2 \
    --epochs 30 \
    --batch_size 8 \
    --cloud_augmentation
```

### Using SLURM (HPC)
```bash
# Submit training job
sbatch scripts/train_cloud_aware_v2.sh

# Monitor progress
squeue -u $USER
tail -f logs/cloud_v2_*.out
```

---

## 💻 Usage Examples

### Inference on New Data
```python
import torch
from models.cloud_aware_fusion import CloudAwareFusionModel

# Load model
model = CloudAwareFusionModel(encoder_name='resnet34', pretrained=False)
checkpoint = torch.load('checkpoints/cloud_aware_v2/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input batch
batch = {
    'optical': optical_tensor,  # (B, 13, H, W)
    'sar': sar_tensor,          # (B, 2, H, W)
    'cloud_mask': cloud_mask    # (B, 1, H, W)
}

# Run inference
with torch.no_grad():
    prediction, gate_weight = model(batch)
    flood_prob = torch.sigmoid(prediction)
    flood_mask = flood_prob > 0.5

print(f"Gate weight (SAR reliance): {gate_weight.item():.2f}")
```

### Custom Training
```python
from dataset_with_clouds import get_cloud_dataloaders
from models.cloud_aware_fusion import get_cloud_aware_model

# Load data with cloud augmentation
train_loader, val_loader, test_loader = get_cloud_dataloaders(
    data_dir="data/sen1flood11",
    batch_size=8,
    cloud_augmentation=True  # Enable synthetic clouds
)

# Create model
model = get_cloud_aware_model(encoder_name='resnet34')

# Train (see train_cloud_aware_2.py for full example)
```

---

## 🔍 Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'rasterio'`**
```bash
conda install -c conda-forge rasterio gdal
```

**Issue: `libnetcdf.so.19: cannot open shared object file`**
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Add to ~/.bashrc to make permanent
```

**Issue: `CUDA out of memory`**
```bash
# Reduce batch size
python train.py --batch_size 4  # Instead of 8
```

**Issue: `AssertionError: S1Hand not found`**
```bash
# Verify data path
python -c "from pathlib import Path; print(Path('data/sen1flood11/data/flood_events/HandLabeled/S1Hand').exists())"

# Update path in scripts if needed
python train.py --data_dir /your/path/to/sen1flood11
```

---

## 🏆 Acknowledgments

- **Dataset:** [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) by Cloud to Street
- **Compute:** UMD HPCC Zaratan cluster
- **Framework:** [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)

---

## 📧 Contact

- **Yatish Sikka** - batman@umd.edu
- **Ritvik Chaturvedi** - ritvikc@umd.edu

**Project Repository:** https://github.com/YatishUMD/FloodVision

---

