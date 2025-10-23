# Flood Vision ETL and Data Preprocessing Pipeline

This repository contains a comprehensive ETL (Extract, Transform, Load) pipeline and data preprocessing utilities for the **FloodVision: Cloud-Aware SAR–Optical Fusion for Global Flood Mapping** project using Sentinel-1 (SAR) and Sentinel-2 (Optical) satellite imagery from the Copernicus Data Space Ecosystem.

## Features

- **Multi-dataset Support**: Handles Sentinel-1/2 data from Copernicus Data Space, SEN12-FLOOD, and SEN1Floods11 datasets
- **SAR Processing**: Speckle filtering, dB conversion, and z-score normalization
- **Optical Processing**: Atmospheric correction, reflectance conversion, and z-score normalization
- **Data Alignment**: Automatic spatial alignment of SAR and optical imagery to 10m resolution
- **Patch Extraction**: 256×256 patches with 50% overlap for comprehensive coverage
- **Metadata Generation**: Automated CSV generation with 100+ samples for baseline experiments
- **Cloud Filtering**: <20% cloud coverage filtering for optimal data quality
- **PyTorch Integration**: Ready-to-use DataLoaders for U-Net and SegFormer models

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YatishUMD/FloodVision
cd FloodVision
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Configure your dataset paths** in `config.json`:

```json
{
  "datasets": {
    "sentinel_data": {
      "path": "./data/sentinel_data",
      "source": "Copernicus Data Space Ecosystem",
      "cloud_coverage_filter": "<20%",
      "time_period": "2021-2022"
    },
    "sen12_flood": {
      "path": "./data/SEN12-FLOOD"
    },
    "sen1floods11": {
      "path": "./data/SEN1Floods11"
    }
  }
}
```

2. **Run the ETL pipeline**:

```python
from etl_pipeline import FloodETLPipeline
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize and run ETL pipeline
pipeline = FloodETLPipeline(config)
pipeline.process_dataset("sentinel_data")
```

3. **Create data loaders**:

```python
from dataset_loaders import create_data_loaders

# Create data loaders for all splits
data_loaders = create_data_loaders(config)

# Use in training loop
for batch in data_loaders['train']:
    sar_data = batch['sar']
    optical_data = batch['optical']
    masks = batch.get('mask', None)
    # ... your training code
```

## File Structure

```
FloodVision/
├── etl_pipeline.py          # Main ETL pipeline
├── data_preprocessing.py    # Preprocessing utilities
├── dataset_loaders.py       # PyTorch dataset classes
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── example_usage.py       # Usage examples
└── README.md              # This file
```

## Configuration

The `config.json` file contains all configuration parameters:

- **Datasets**: Paths and settings for each dataset
- **Processing**: Image processing parameters (resolution, patch size, normalization)
- **Training**: Model training parameters
- **Data Splits**: Train/validation/test split ratios
- **Output**: Paths for processed data and results

## Key Components

### 1. ETL Pipeline (`etl_pipeline.py`)

The main ETL pipeline handles:

- Data extraction from various formats
- SAR and optical imagery preprocessing
- Spatial alignment and resampling
- Patch creation and storage

### 2. Data Preprocessing (`data_preprocessing.py`)

Comprehensive preprocessing utilities:

- **ImagePreprocessor**: SAR and optical data preprocessing
- **DataAugmentation**: Training data augmentation
- **DataAlignment**: Spatial alignment utilities
- **DataNormalization**: Various normalization strategies

### 3. Dataset Loaders (`dataset_loaders.py`)

PyTorch-compatible dataset classes:

- **SEN12MSDataset**: SEN12MS-CR dataset loader
- **SEN12FloodDataset**: SEN12-FLOOD dataset loader
- **SEN1Floods11Dataset**: SEN1Floods11 dataset loader
- **CombinedFloodDataset**: Multi-dataset combination

## Usage Examples

### Basic ETL Processing

```python
from etl_pipeline import FloodETLPipeline

pipeline = FloodETLPipeline("config.json")
pipeline.process_dataset("sentinel_data")
```

### Custom Preprocessing

```python
from data_preprocessing import create_preprocessing_pipeline

pipeline = create_preprocessing_pipeline(config['processing'])

# Preprocess SAR data
processed_sar = pipeline['preprocessor'].preprocess_sar(sar_data)

# Apply augmentation
aug_sar, aug_optical, aug_mask = pipeline['augmentation'].augment_data(
    sar_data, optical_data, mask_data
)
```

### Training Data Loading

```python
from dataset_loaders import create_data_loaders

data_loaders = create_data_loaders(config)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loaders['train']:
        # Your training code here
        pass
```

## Dataset Requirements

### Expected Directory Structure

```
data/
├── SEN12MS-CR/
│   ├── sar/
│   │   ├── image1.tif
│   │   └── image2.tif
│   ├── optical/
│   │   ├── image1.tif
│   │   └── image2.tif
│   └── masks/
│       ├── image1.tif
│       └── image2.tif
├── SEN12-FLOOD/
│   ├── sar/
│   ├── optical/
│   └── flood_masks/
└── SEN1Floods11/
    ├── sar/
    ├── optical/
    └── flood_masks/
```

### Data Format Requirements

- **SAR Data**: GeoTIFF format with VV and VH bands
- **Optical Data**: GeoTIFF format with Sentinel-2 bands (B2, B3, B4, B8, B11, B12)
- **Masks**: Single-band GeoTIFF with flood/no-flood labels

## Advanced Features

### Custom Normalization

```python
from data_preprocessing import DataNormalization

# Compute dataset statistics
mean, std = DataNormalization.compute_dataset_statistics(data_list)

# Normalize data
normalized_data = DataNormalization.normalize_by_dataset_statistics(
    data, mean, std
)
```

### Patch-based Processing

```python
from data_preprocessing import DataAlignment

aligner = DataAlignment(config['processing'])
patches = aligner.create_aligned_patches(
    sar_data, optical_data,
    patch_size=256, overlap=0.1
)
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`
2. **Path Issues**: Ensure dataset paths in `config.json` are correct
3. **Memory Issues**: Reduce batch size or patch size in configuration
4. **File Format Issues**: Ensure all data files are in GeoTIFF format

### Performance Optimization

- Use `num_workers > 0` for parallel data loading
- Enable `pin_memory=True` for GPU training
- Use mixed precision training when available
- Consider data caching for frequently accessed datasets
