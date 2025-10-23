"""
Example usage of the Flood Vision ETL and Data Preprocessing Pipeline
"""

import json
import logging
from etl_pipeline import FloodETLPipeline
from dataset_loaders import create_data_loaders, get_dataset_statistics
from data_preprocessing import create_preprocessing_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating ETL and preprocessing usage"""
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    logger.info("Starting Flood Vision ETL and Preprocessing Pipeline")
    
    # Step 1: Run ETL Pipeline
    logger.info("Step 1: Running ETL Pipeline")
    etl_pipeline = FloodETLPipeline(config)
    
    # Process each dataset
    datasets_to_process = ["sen12ms_cr", "sen12_flood", "sen1floods11"]
    
    for dataset_name in datasets_to_process:
        logger.info(f"Processing dataset: {dataset_name}")
        success = etl_pipeline.process_dataset(dataset_name)
        
        if success:
            logger.info(f"Successfully processed {dataset_name}")
        else:
            logger.warning(f"Failed to process {dataset_name} or dataset not found")
    
    # Step 2: Create Data Loaders
    logger.info("Step 2: Creating Data Loaders")
    data_loaders = create_data_loaders(config)
    
    # Step 3: Compute Dataset Statistics
    logger.info("Step 3: Computing Dataset Statistics")
    if 'train' in data_loaders:
        train_stats = get_dataset_statistics(data_loaders['train'])
        
        logger.info("SAR Statistics:")
        logger.info(f"  Mean: {train_stats['sar']['mean']}")
        logger.info(f"  Std: {train_stats['sar']['std']}")
        
        logger.info("Optical Statistics:")
        logger.info(f"  Mean: {train_stats['optical']['mean']}")
        logger.info(f"  Std: {train_stats['optical']['std']}")
        
        # Save statistics for later use
        with open('dataset_statistics.json', 'w') as f:
            json.dump({
                'sar_mean': train_stats['sar']['mean'].tolist(),
                'sar_std': train_stats['sar']['std'].tolist(),
                'optical_mean': train_stats['optical']['mean'].tolist(),
                'optical_std': train_stats['optical']['std'].tolist()
            }, f, indent=2)
    
    # Step 4: Test Data Loading
    logger.info("Step 4: Testing Data Loading")
    
    for split, data_loader in data_loaders.items():
        logger.info(f"\n{split.upper()} DataLoader:")
        logger.info(f"  Number of batches: {len(data_loader)}")
        
        # Get a sample batch
        for batch_idx, batch in enumerate(data_loader):
            logger.info(f"  Batch {batch_idx + 1}:")
            logger.info(f"    SAR shape: {batch['sar'].shape}")
            logger.info(f"    Optical shape: {batch['optical'].shape}")
            
            if 'mask' in batch:
                logger.info(f"    Mask shape: {batch['mask'].shape}")
            
            # Only show first batch for each split
            break
    
    # Step 5: Demonstrate Preprocessing Pipeline
    logger.info("Step 5: Demonstrating Preprocessing Pipeline")
    
    preprocessing_pipeline = create_preprocessing_pipeline(config['processing'])
    
    # Example preprocessing
    import numpy as np
    
    # Create dummy data for demonstration
    dummy_sar = np.random.rand(2, 256, 256) * 100  # Simulate SAR data
    dummy_optical = np.random.rand(6, 256, 256) * 10000  # Simulate optical data
    
    # Preprocess SAR data
    processed_sar = preprocessing_pipeline['preprocessor'].preprocess_sar(dummy_sar)
    logger.info(f"SAR preprocessing: {dummy_sar.shape} -> {processed_sar.shape}")
    
    # Preprocess optical data
    processed_optical = preprocessing_pipeline['preprocessor'].preprocess_optical(dummy_optical)
    logger.info(f"Optical preprocessing: {dummy_optical.shape} -> {processed_optical.shape}")
    
    # Apply augmentation
    aug_sar, aug_optical, _ = preprocessing_pipeline['augmentation'].augment_data(
        processed_sar, processed_optical, None
    )
    logger.info(f"Augmentation applied successfully")
    
    logger.info("ETL and Preprocessing Pipeline completed successfully!")

if __name__ == "__main__":
    main()
