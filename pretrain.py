import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CloudSEN12Dataset
from src.models.pretraining import CloudNet

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Pre-training")
    for batch in pbar:
        # CloudSEN12Dataset returns {'s2': ..., 'label': ...}
        img = batch['s2'].to(device)
        label = batch['label'].to(device)
        
        # Squeeze label channel if needed (B, 1, H, W) -> (B, H, W)
        if len(label.shape) == 4:
            label = label.squeeze(1)
            
        # Forward
        outputs = model(img)
        loss = criterion(outputs, label)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", default="checkpoints/pretrained_s2.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 1. Dataset
    # We use all available data for pre-training (or split if you want val)
    ds = CloudSEN12Dataset(split="train")
    if len(ds) == 0:
        print("Dataset empty. Exiting.")
        return
        
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # 2. Model
    model = CloudNet(num_classes=2).to(device)
    
    # 3. Optimize
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting Pre-training on {len(ds)} samples...")
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
        
        # Save weights
        torch.save(model.encoder.state_dict(), args.save_path)
        print(f"Saved encoder weights to {args.save_path}")

if __name__ == "__main__":
    main()

