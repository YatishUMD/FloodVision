# FloodVision: Multi-Modal Flood Detection

FloodVision is a deep learning project designed to detect flood water from satellite imagery. It leverages multi-modal data fusion, combining Sentinel-1 (SAR) and Sentinel-2 (Optical) imagery to robustly map floods even in cloudy conditions where optical sensors fail.

## Project Overview

Traditional flood mapping relies heavily on optical imagery, which is often obstructed by clouds during storm events. Radar (SAR) sees through clouds but is noisy and harder to interpret. This project uses a **Gated Fusion Network** to:

1.  **Pre-train** on massive cloud datasets to learn robust optical features.
2.  **Fuse** Optical and Radar data using an attention mechanism that learns to "trust" the Radar signal when the Optical signal is cloudy.
3.  **Detect** water/flood pixels with high accuracy across varying weather conditions.

## Repository Structure

```
flood vision/
├── checkpoints/          # Saved model weights
├── data/                 # Dataset storage
│   ├── sen1floods11/     # Main flood dataset
│   └── cloudsen12/       # Pre-training cloud dataset
├── src/
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset classes (Sen1Floods11, CloudSEN12)
│   │   ├── download_datasets.py  # Scripts to fetch data from cloud/FTP
│   │   └── preprocessing.py      # Splits data into Train/Test (Geographic Split)
│   ├── models/
│   │   ├── baselines.py          # Baseline models (Optical, SAR, Simple Fusion)
│   │   ├── decoders.py           # Main FloodNet architecture
│   │   ├── fusion.py             # Gated Fusion module (Attention mechanism)
│   │   └── pretraining.py        # CloudNet for pre-training
│   ├── training/
│       └── train.py              # Main training script (Flood Detection)
├── evaluate.py           # Evaluation script (IoU metrics)
├── pretrain.py           # Pre-training script (Cloud Detection)
├── visualize.py          # Visualization script (Comparisons & Attention Maps)
└── notes.txt             # Development notes
```

## Getting Started

### 1. Prerequisites

- Python 3.8+
- PyTorch
- Rasterio
- Albumentations
- Google Cloud SDK (`gsutil`) - for downloading Sen1Floods11
- Matplotlib (for visualization)

### 2. Data Preparation

This project automates data downloading. You will need approximately **15GB** of space for the core datasets.

**Step A: Download Data**

```bash
# Download Sen1Floods11 (Flood Data)
python src/data/download_datasets.py sen1floods11

# Download CloudSEN12 (Cloud Data for Pre-training)
python src/data/download_datasets.py cloudsen12
```

**Step B: Generate Splits**
To ensure fair evaluation, we split the data geographically (e.g., training on America/Africa, testing on Asia).

```bash
# Scans the downloaded files and creates train_split.csv / test_split.csv
python src/data/preprocessing.py
```

## Model Training

### Phase 1: Pre-training (Optional but Recommended)

Pre-train the Optical Encoder to understand clouds. This helps the fusion module identify when to ignore the optical signal.

```bash
python pretrain.py --epochs 5 --save_path checkpoints/pretrained_s2.pth
```

### Phase 2: Baselines & Flood Detection Training

You can train various model architectures including baselines and the main fusion model.

**Train Baselines:**

```bash
# Optical-only Baseline (Sentinel-2)
python src/training/train.py --model_type optical --epochs 50 --batch_size 16 --experiment_name optical_baseline

# SAR-only Baseline (Sentinel-1)
python src/training/train.py --model_type sar --epochs 50 --batch_size 16 --experiment_name sar_baseline

# Simple Fusion Baseline (Concatenation)
python src/training/train.py --model_type simple --epochs 50 --batch_size 16 --experiment_name simple_fusion
```

**Train Main Gated Fusion Model:**
Train the full dual-stream model (S1 + S2) with the gated attention mechanism. If you ran Phase 1, you can load the pre-trained weights.

```bash
python src/training/train.py --model_type gated --epochs 50 --batch_size 16 --s2_weights checkpoints/pretrained_s2.pth --experiment_name gated_fusion
```

- **Outputs:** Saves the best model to `checkpoints/{experiment_name}_best.pth`.
- **Logs:** Displays training loss and validation IoU (Intersection over Union) for the Water class.

## Evaluation

Evaluate the model on the unseen test regions. The evaluation script calculates the **Mean IoU** and also provides a stratified report (performance on clear vs. cloudy images).

```bash
# Evaluate Optical Baseline
python evaluate.py --model_type optical --checkpoint checkpoints/optical_baseline_best.pth

# Evaluate Gated Fusion Model
python evaluate.py --model_type gated --checkpoint checkpoints/gated_fusion_best.pth
```

**Example Output:**

```
--- Evaluation Results ---
Overall mIoU: 0.7421
Clear Images (n=45) mIoU:  0.7810
Cloudy Images (n=12) mIoU: 0.6105
```

## Visualization

You can generate visualizations to interpret the model's behavior, specifically the **Gated Attention Map** which shows where the model relies on Optical vs. SAR data.

```bash
python visualize.py --checkpoint checkpoints/gated_fusion_best.pth --num_samples 10
```

Images will be saved to the `visualizations/` folder, showing:

1.  Optical Image (RGB)
2.  SAR Image (Radar)
3.  **Attention Map** (Dark = Trust Radar, Bright = Trust Optical)
4.  Ground Truth Mask
5.  Model Prediction

## Model Architecture

The core model (`FloodNet`) consists of:

1.  **S1 Encoder:** ResNet-based encoder for SAR (Sentinel-1) data.
2.  **S2 Encoder:** ResNet-based encoder for Optical (Sentinel-2) data.
3.  **Gated Fusion Module:** A learnable attention gate that takes features from both encoders and outputs a weighted combination:
    $F_{fused} = \alpha \cdot F_{S2} + (1 - \alpha) \cdot F_{S1}$
    Where $\alpha$ is the learned attention map (0 to 1).
4.  **Decoder:** Upsamples the fused features to generate the final binary flood mask.
