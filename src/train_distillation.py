import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import yaml
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project Imports
from models.student import StudentModel
from models.teacher import Teacher
from utils.data_loader import PolypDataset
from utils.losses import DiceBCELoss, KDLoss
from utils.metrics import calculate_metrics, AverageMeter

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_transforms(img_size, mode="train"):
    """Returns Albumentations transforms for training or validation."""
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

def train_one_epoch(student, teacher, loader, optimizer, loss_fn_sup, loss_fn_kd, scaler, device, config):
    student.train()
    total_losses = AverageMeter()
    sup_losses = AverageMeter()
    kd_losses = AverageMeter()
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        student_logits = student(images)
        
        with torch.no_grad():
            teacher_logits = teacher(images)

        if torch.isnan(student_logits).any() or torch.isnan(teacher_logits).any():
            continue

        loss_sup = loss_fn_sup(student_logits, masks)
        loss_kd = loss_fn_kd(student_logits, teacher_logits)
        
        alpha = config['alpha']
        total_loss = alpha * loss_sup + (1 - alpha) * loss_kd

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        
        if not torch.isnan(total_loss):
            total_losses.update(total_loss.item(), images.size(0))
            sup_losses.update(loss_sup.item(), images.size(0))
            kd_losses.update(loss_kd.item(), images.size(0))
            pbar.set_postfix({"Loss": f"{total_losses.avg:.4f}", "Sup": f"{sup_losses.avg:.4f}"})
        
    return total_losses.avg, sup_losses.avg, kd_losses.avg

def validate(student, loader, device):
    student.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = student(images)
            metrics = calculate_metrics(outputs, masks)
            
            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))
            
    return dice_meter.avg, iou_meter.avg

def main():
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training on {device} ---")
    
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

    print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)}")

    print("Initializing Models...")
    student = StudentModel(encoder_name=config['student_backbone']).to(device)
    teacher = Teacher(checkpoint_path=config['teacher_checkpoint'], device=device)

    loss_fn_sup = DiceBCELoss().to(device)
    loss_fn_kd = KDLoss(temperature=config['temperature']).to(device)
    
    optimizer = optim.AdamW(student.parameters(), lr=config['learning_rate'])
    scaler = None

    best_dice = 0.0
    
    # Training History
    history = {
        'model_name': 'Distillation',
        'epochs': [],
        'train_loss': [],
        'train_loss_sup': [],
        'train_loss_kd': [],
        'val_dice': [],
        'val_iou': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        
        # Train
        train_loss, train_loss_sup, train_loss_kd = train_one_epoch(
            student, teacher, train_loader, optimizer, loss_fn_sup, loss_fn_kd, scaler, device, config
        )
        
        # Validate
        val_dice, val_iou = validate(student, val_loader, device)
        
        print(f"Loss: {train_loss:.4f} (Sup: {train_loss_sup:.4f}, KD: {train_loss_kd:.4f}) | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        # Save to history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_loss_sup'].append(train_loss_sup)
        history['train_loss_kd'].append(train_loss_kd)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config['save_dir'], f"{config['experiment_name']}_best.pth")
            torch.save(student.state_dict(), save_path)
            print(f"--> Best Model Saved! (Dice: {best_dice:.4f})")
    
    # Save training history
    history_path = os.path.join(config['save_dir'], f"{config['experiment_name']}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()