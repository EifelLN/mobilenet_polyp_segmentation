import os
import yaml
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Reuse existing modules
from models.student import StudentModel
from utils.data_loader import PolypDataset
from utils.losses import DiceBCELoss
from utils.metrics import calculate_metrics, AverageMeter

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_transforms(img_size, mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomGamma(p=0.2),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
        ])

def train_one_epoch(student, loader, optimizer, loss_fn, device):
    student.train()
    losses = AverageMeter()
    
    pbar = tqdm(loader, desc="Training (Baseline)", leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = student(images)
        loss = loss_fn(logits, masks)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}"})
        
    return losses.avg

def validate(student, loader, device):
    student.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = student(images)
            metrics = calculate_metrics(logits, masks)
            
            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))
            
    return dice_meter.avg, iou_meter.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting BASELINE Training on {device} ---")
    
    exp_name = f"{config['experiment_name']}_BASELINE"
    os.makedirs(config['save_dir'], exist_ok=True)

    train_dataset = PolypDataset(
        root_dir=os.path.join(config['data_root'], 'TrainDataset'),
        transform=get_transforms(config['img_size'], mode="train"),
        img_size=config['img_size']
    )
    val_dataset = PolypDataset(
        root_dir=os.path.join(config['data_root'], 'TestDataset', 'CVC-ClinicDB'), 
        transform=get_transforms(config['img_size'], mode="val"),
        img_size=config['img_size']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )

    student = StudentModel(encoder_name=config['student_backbone']).to(device)

    loss_fn = DiceBCELoss().to(device)
    optimizer = optim.AdamW(student.parameters(), lr=config['learning_rate'])

    best_dice = 0.0
    
    # Training History
    history = {
        'model_name': 'Baseline',
        'epochs': [],
        'train_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        
        train_loss = train_one_epoch(student, train_loader, optimizer, loss_fn, device)
        val_dice, val_iou = validate(student, val_loader, device)
        
        print(f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        # Save to history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config['save_dir'], f"{exp_name}_best.pth")
            torch.save(student.state_dict(), save_path)
            print(f"--> Best Baseline Saved! (Dice: {best_dice:.4f})")
    
    # Save training history
    history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main()