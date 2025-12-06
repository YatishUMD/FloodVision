import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.data.dataset import Sen1Floods11Dataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.decoders import FloodNet
from src.models.baselines import OpticalUNet, SarUNet, SimpleFusionUNet

def get_model(model_type, num_classes=2, s2_weights_path=None):
    if model_type == 'optical':
        return OpticalUNet(num_classes=num_classes)
    elif model_type == 'sar':
        return SarUNet(num_classes=num_classes)
    elif model_type == 'simple':
        return SimpleFusionUNet(num_classes=num_classes)
    elif model_type == 'gated':
        return FloodNet(num_classes=num_classes, s2_weights_path=s2_weights_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def calculate_iou(pred, label):
    pred = torch.argmax(pred, dim=1).view(-1)
    label = label.view(-1)
    
    intersection = ((pred == 1) & (label == 1)).sum().item()
    union = ((pred == 1) | (label == 1)).sum().item()
    
    return intersection, union

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        s1 = batch['s1'].to(device)
        s2 = batch['s2'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(s1, s2)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_inter = 0
    total_union = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            s1 = batch['s1'].to(device)
            s2 = batch['s2'].to(device)
            label = batch['label'].to(device)
            
            outputs = model(s1, s2)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            
            inter, union = calculate_iou(outputs, label)
            total_inter += inter
            total_union += union
            
    iou = total_inter / (total_union + 1e-6)
    return running_loss / len(loader), iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=['optical', 'sar', 'simple', 'gated'],
                        help="Model architecture to train")
    parser.add_argument("--data_dir", default="data/sen1floods11", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--s2_weights", default=None, help="Path to pretrained S2 encoder weights")
    parser.add_argument("--save_dir", default="checkpoints", help="Directory to save models")
    parser.add_argument("--experiment_name", default=None, help="Name for the experiment/file")
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        args.experiment_name = args.model_type
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    
    # Data
    train_ds = Sen1Floods11Dataset(args.data_dir, split="train", transform=get_train_transforms())
    val_ds = Sen1Floods11Dataset(args.data_dir, split="test", transform=get_val_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = get_model(args.model_type, s2_weights_path=args.s2_weights).to(device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss() # Can add weight=torch.tensor([1.0, 5.0]) for balance
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_iou = 0.0
    
    print(f"Starting training for {args.model_type} model...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        scheduler.step(val_iou)
        
        if val_iou > best_iou:
            best_iou = val_iou
            save_path = os.path.join(args.save_dir, f"{args.experiment_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            
    print(f"Training complete. Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
