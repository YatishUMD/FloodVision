import os
import pandas as pd
import glob
from pathlib import Path

def create_splits(data_dir, output_dir=None):
    """
    Generates train/test splits based on geographic regions to ensure zero overlap.
    
    Args:
        data_dir (str): Path to 'HandLabeled' directory.
        output_dir (str, optional): Where to save .csv files. Defaults to data_dir's parent.
    """
    
    # Define Geographic Split Strategy
    # Test Set = Asia
    # Train Set = Americas, Africa, Europe, etc.
    TEST_REGIONS = ["India", "Cambodia", "Pakistan", "Sri-Lanka", "Mekong", "Myanmar"]
    
    # Find all Label files (Ground Truth)
    # Expected path structure: .../HandLabeled/LabelHand/*.tif
    label_dir = os.path.join(data_dir, "LabelHand")
    
    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found at {label_dir}")
        return

    print(f"Scanning {label_dir}...")
    label_files = glob.glob(os.path.join(label_dir, "*.tif"))
    
    data = []
    for filepath in label_files:
        filename = os.path.basename(filepath)
        # Filename format: Region_ID_LabelHand.tif (e.g., Bolivia_123456_LabelHand.tif)
        # We want the ID: Region_ID_LabelHand (matches Dataset class expectation)
        file_id = filename.replace(".tif", "")
        
        # Extract region name (everything before the first underscore)
        region = file_id.split("_")[0]
        
        # Determine split
        if region in TEST_REGIONS:
            split = "test"
        else:
            split = "train"
            
        data.append({
            "id": file_id,
            "region": region,
            "split": split
        })
    
    df = pd.DataFrame(data)
    
    # Verify we have data
    if len(df) == 0:
        print("No files found! Check data path.")
        return

    # Print stats
    print("\nSplit Statistics:")
    print(df["split"].value_counts())
    print("\nRegions in Test Set:", df[df["split"]=="test"]["region"].unique())
    print("Regions in Train Set:", df[df["split"]=="train"]["region"].unique())
    
    # Save splits
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(data_dir)) # Up 2 levels to data/sen1floods11/v1.1/data/
        
    # We'll save them in the project root data/sen1floods11 folder for easier access
    # Or simply where the user asked. Let's aim for 'data/sen1floods11/'
    # Assuming data_dir input is '.../HandLabeled'
    
    # Let's save to a standard location: src/data/splits/ (better for repro) 
    # OR data/sen1floods11/splits/
    
    save_dir = os.path.join(output_dir, "splits")
    os.makedirs(save_dir, exist_ok=True)
    
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    
    train_csv = os.path.join(save_dir, "train_split.csv")
    test_csv = os.path.join(save_dir, "test_split.csv")
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\nSaved splits to {save_dir}")
    print(f"Train: {len(train_df)} samples")
    print(f"Test:  {len(test_df)} samples")

if __name__ == "__main__":
    # Auto-detect path if run as script
    # Assumes running from project root or src/data
    
    # Try to find the downloaded data
    possible_paths = [
        "data/sen1floods11/v1.1/data/flood_events/HandLabeled",
        "../../data/sen1floods11/v1.1/data/flood_events/HandLabeled"
    ]
    
    target_path = None
    for p in possible_paths:
        if os.path.exists(p):
            target_path = p
            break
            
    if target_path:
        # Save splits to 'data/sen1floods11' root
        output_base = target_path.split("v1.1")[0] # .../data/sen1floods11/
        create_splits(target_path, output_dir=output_base)
    else:
        print("Could not find Sen1Floods11 data. Please run download_datasets.py first.")

