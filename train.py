import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import our modules
from src.data.dataset import Sen1Floods11Dataset
from src.models.decoders import FloodNet
from src.data.transforms import get_train_transforms, get_val_transforms

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # 1. Load data to device
        s1 = batch['s1'].to(device)
        s2 = batch['s2'].to(device)
        label = batch['label'].to(device) # Shape: (B, 512, 512)
        
        # 2. Forward pass
        outputs = model(s1, s2) # Shape: (B, 2, 512, 512)
        
        # 3. Calculate Loss
        loss = criterion(outputs, label)
        
        # 4. Backward & Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. Stats
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    # Simple IoU tracker
    intersection = 0
    union = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for batch in pbar:
            s1 = batch['s1'].to(device)
            s2 = batch['s2'].to(device)
            label = batch['label'].to(device)
            
            outputs = model(s1, s2)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            
            # Calculate simple metrics (Water Class = 1)
            preds = torch.argmax(outputs, dim=1) # (B, H, W)
            
            # IoU for Class 1 (Water)
            # Flatten
            pred_flat = preds.view(-1)
            label_flat = label.view(-1)
            
            # Intersection: Pred=1 AND Label=1
            intersection += ((pred_flat == 1) & (label_flat == 1)).sum().item()
            # Union: Pred=1 OR Label=1
            union += ((pred_flat == 1) | (label_flat == 1)).sum().item()
            
    epoch_loss = running_loss / len(loader)
    iou = intersection / (union + 1e-6) # Avoid div by zero
    
    return epoch_loss, iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/sen1floods11", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--s2_weights", default="checkpoints/pretrained_s2.pth", help="Path to pre-trained S2 weights")
    args = parser.parse_args()
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps") # Mac M1/M2/M3 Acceleration
    print(f"Using device: {device}")
    
    # Create Checkpoint Dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Datasets
    train_ds = Sen1Floods11Dataset(args.data_dir, split="train", transform=get_train_transforms())
    test_ds = Sen1Floods11Dataset(args.data_dir, split="test", transform=get_val_transforms())
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 2. Model
    # Try to load pre-trained S2 weights if available
    s2_weights = args.s2_weights if os.path.exists(args.s2_weights) else None
    model = FloodNet(num_classes=2, s2_weights_path=s2_weights).to(device)
    
    # 3. Optimizer & Loss
    # Water is often rare compared to land, so we might need class weights later.
    # For now, standard CrossEntropy.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    best_iou = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val IoU (Water): {val_iou:.4f}")
        
        # Save Best Model
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")
            
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()

