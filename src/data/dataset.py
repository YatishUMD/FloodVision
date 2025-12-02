import os
import torch
import rasterio
import numpy as np
import tacoreader
from torch.utils.data import Dataset

class Sen1Floods11Dataset(Dataset):
    """
    Dataset class for Sen1Floods11 using HandLabeled data.
    Loads S1 (SAR), S2 (Optical), and Label (Water Mask).
    """
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to root data directory (e.g., 'data/sen1floods11')
            split (str): 'train' or 'test' (loads from generated csv splits)
            transform (callable, optional): Albumentations transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # paths assuming standard Sen1Floods11 structure
        # v1.1/data/flood_events/HandLabeled/
        self.base_path = os.path.join(root_dir, "v1.1", "data", "flood_events", "HandLabeled")
        self.s1_dir = os.path.join(self.base_path, "S1Hand")
        self.s2_dir = os.path.join(self.base_path, "S2Hand")
        self.label_dir = os.path.join(self.base_path, "LabelHand")
        
        # load file IDs from CSV splits if they exist
        split_dir = os.path.join(root_dir, "splits")
        csv_path = os.path.join(split_dir, f"{split}_split.csv")
        
        if os.path.exists(csv_path):
            print(f"Loading {split} split from {csv_path}")
            df = pd.read_csv(csv_path)
            # use the 'id' column which contains filenames like 'Region_ID_LabelHand'
            self.file_ids = df["id"].tolist()
        else:
            # fallback logic...
            if os.path.exists(self.label_dir):
                all_files = sorted([f.replace(".tif", "") for f in os.listdir(self.label_dir) if f.endswith(".tif")])
                split_idx = int(0.8 * len(all_files))
                if split == "train":
                    self.file_ids = all_files[:split_idx]
                else:
                    self.file_ids = all_files[split_idx:]
            else:
                self.file_ids = []

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        region_id = file_id.replace("_LabelHand", "")
        
        s1_path = os.path.join(self.s1_dir, f"{region_id}_S1Hand.tif")
        s2_path = os.path.join(self.s2_dir, f"{region_id}_S2Hand.tif")
        label_path = os.path.join(self.label_dir, f"{file_id}.tif")
        
        s1 = self._load_tif(s1_path)      # Shape: (C, H, W)
        s2 = self._load_tif(s2_path)      # Shape: (C, H, W)
        label = self._load_tif(label_path) # Shape: (1, H, W) usually
        
        s1 = self._preprocess_s1(s1)
        s2 = self._preprocess_s2(s2)
        
        data = {
            "image": np.moveaxis(s2, 0, -1),   # treat S2 as main image
            "s1": np.moveaxis(s1, 0, -1),      # pass S1 as extra
            "mask": label.squeeze()            # (H, W)
        }
        
        if self.transform:
            augmented = self.transform(**data)
            s2 = augmented["image"]
            s1 = augmented["s1"]
            label = augmented["mask"]
            
            s2 = torch.from_numpy(np.moveaxis(s2, -1, 0)).float()
            s1 = torch.from_numpy(np.moveaxis(s1, -1, 0)).float()
            label = torch.from_numpy(label).long()
        else:
            s2 = torch.from_numpy(s2).float()
            s1 = torch.from_numpy(s1).float()
            if len(label.shape) == 3:
                label = label.squeeze(0)
            label = torch.from_numpy(label).long()

        return {
            "s1": s1, 
            "s2": s2, 
            "label": label,
            "id": region_id
        }

    def _load_tif(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with rasterio.open(path) as src:
            image = src.read()
        return image.astype(np.float32)

    def _preprocess_s1(self, img):
        img = np.clip(img, -25, 0)
        img = (img + 25) / 25 
        return img

    def _preprocess_s2(self, img):
        img = img / 10000.0
        img = np.clip(img, 0, 1)
        return img


class CloudSEN12Dataset(Dataset):
    """
    Dataset for Pre-training on CloudSEN12.
    Task: Input S2 (Optical) -> Predict Cloud Mask.
    """
    def __init__(self, root_dir="data/cloudsen12", split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Find .taco files
        taco_files = sorted([
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith(".taco")
        ])
        
        if not taco_files:
            print(f"Warning: No .taco files found in {root_dir}")
            self.dataset = None
            self.length = 0
        else:
            print(f"Loading CloudSEN12 from {taco_files}...")
            self.dataset = tacoreader.load(taco_files)
            # Total length is hard to get without iterating, but tacoreader might support len()
            # If not, we might need to cache indices. 
            # For now, let's assume len() works or we limit it.
            try:
                self.length = len(self.dataset)
            except:
                self.length = 1000 # Fallback limit if len() fails
            
            # Simple Split Logic (First 80% train, last 20% val)
            split_idx = int(0.8 * self.length)
            if split == "train":
                self.indices = range(0, split_idx)
            else:
                self.indices = range(split_idx, self.length)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Read from tacoreader
        # 0 = L1C (Optical), 1 = Label (Cloud Mask)
        img_handle = self.dataset.read(real_idx).read(0)
        lbl_handle = self.dataset.read(real_idx).read(1)
        
        # Load data using rasterio
        with rasterio.open(img_handle) as src:
            image = src.read().astype(np.float32) # (C, H, W)
            
        with rasterio.open(lbl_handle) as src:
            mask = src.read(1).astype(np.float32) # (H, W)
            
        # Preprocess Image (S2 normalization)
        image = image / 10000.0
        image = np.clip(image, 0, 1)
        
        # Preprocess Mask
        # Original: 0=Clear, 1=Thick, 2=Thin, 3=Shadow
        # Target: 0=Clear, 1=Cloud (Merge 1, 2, 3 into 1 for binary segmentation)
        # Or keep classes? Let's do binary Cloud/NoCloud
        binary_mask = (mask > 0).astype(np.float32)
        
        if self.transform:
            # Albumentations expects HWC
            data = {"image": np.moveaxis(image, 0, -1), "mask": binary_mask}
            augmented = self.transform(**data)
            image = augmented["image"]
            mask = augmented["mask"]
            
            # Back to CHW
            image = torch.from_numpy(np.moveaxis(image, -1, 0)).float()
            mask = torch.from_numpy(mask).long()
        else:
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(binary_mask).long()
            
        return {
            "s2": image,
            "label": mask
        }
