import sys
sys.path.append('/home/batman/msml640/FloodVision')

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import json

from dataset_with_clouds import get_cloud_dataloaders
from models.cloud_aware_fusion import get_cloud_aware_model

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    if isinstance(pred, tuple):
        pred, gate = pred
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def calculate_iou(pred, target, threshold=0.5):
    if isinstance(pred, tuple):
        pred = pred[0]
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    num_valid = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        pred, gate = model(batch)
        loss = combined_loss(pred, batch['label'])
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_iou += calculate_iou(pred, batch['label'])
        num_valid += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_valid, total_iou / num_valid

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            pred, gate = model(batch)
            loss = combined_loss(pred, batch['label'])
            
            total_loss += loss.item()
            total_iou += calculate_iou(pred, batch['label'])
    
    return total_loss / len(dataloader), total_iou / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Cloud augmentation: {args.cloud_augmentation}\n")
    
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders with cloud augmentation
    print("Loading data...")
    train_loader, val_loader, test_loader = get_cloud_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cloud_augmentation=args.cloud_augmentation
    )
    
    # Create model
    print("Creating cloud-aware fusion model...")
    model = get_cloud_aware_model(encoder_name=args.encoder, pretrained=False)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    best_iou = 0
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        
        val_loss, val_iou = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/train', train_iou, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)
        
        scheduler.step(val_iou)
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_iou': val_iou,
            }, output_dir / 'best_model.pth')
            print(f"✓ Saved best model (IoU: {val_iou:.4f})")
    
    print(f"\nTraining complete! Best Val IoU: {best_iou:.4f}")
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='/home/batman/scratch.msml640/data/sen1flood11')
    parser.add_argument('--output_dir', type=str, 
                       default='~/scratch.msml640/checkpoints/cloud_aware_v2')
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cloud_augmentation', action='store_true', default=True)
    
    args = parser.parse_args()
    main(args)
