"""
Unified Training Script for Polyp Segmentation
===============================================
Supports three training modes:
1. baseline    - Train student model from scratch (no teacher)
2. distillation - Train student with Knowledge Distillation from teacher
3. teacher     - Train the teacher model from scratch
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import yaml
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Project Imports
from models.student import StudentModel
from models.teacher import Teacher
from utils.data_loader import PolypDataset
from utils.losses import DiceBCELoss, KDLoss
from utils.metrics import calculate_metrics, AverageMeter


def load_config(config_path):
    """Load configuration from YAML file."""
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


def create_dataloaders(config):
    """Create train and validation dataloaders."""
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
    return train_loader, val_loader


# ============== Training Functions ==============

def train_baseline_epoch(model, loader, optimizer, loss_fn, device):
    """Train one epoch for baseline or teacher model (supervised only)."""
    model.train()
    losses = AverageMeter()

    pbar = tqdm(loader, desc="Training", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({"Loss": f"{losses.avg:.4f}"})

    return losses.avg


def train_distillation_epoch(student, teacher, loader, optimizer, loss_fn_sup, loss_fn_kd, device, config):
    """Train one epoch with Knowledge Distillation."""
    student.train()
    losses = AverageMeter()

    pbar = tqdm(loader, desc="Training (KD)", leave=False)

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
            losses.update(total_loss.item(), images.size(0))
            pbar.set_postfix({"Loss": f"{losses.avg:.4f}"})

    return losses.avg


def validate(model, loader, device):
    """Validate model on validation set."""
    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            metrics = calculate_metrics(logits, masks)

            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))

    return dice_meter.avg, iou_meter.avg


# ============== Main Training Logic ==============

def train_baseline(config, device):
    """Train student model from scratch (baseline - no teacher)."""
    print(f"--- Starting BASELINE Training on {device} ---")

    exp_name = f"{config['experiment_name']}_BASELINE"
    os.makedirs(config['save_dir'], exist_ok=True)

    train_loader, val_loader = create_dataloaders(config)

    # Initialize student model
    student = StudentModel(encoder_name=config['student_backbone']).to(device)

    loss_fn = DiceBCELoss().to(device)
    optimizer = optim.AdamW(student.parameters(), lr=config['learning_rate'])

    best_dice = 0.0
    history = {
        'model_name': 'Baseline',
        'epochs': [],
        'train_loss': [],
        'val_dice': [],
        'val_iou': []
    }

    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch + 1}/{config['epochs']}]")

        train_loss = train_baseline_epoch(student, train_loader, optimizer, loss_fn, device)
        val_dice, val_iou = validate(student, val_loader, device)

        print(f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config['save_dir'], f"{exp_name}_best.pth")
            torch.save(student.state_dict(), save_path)
            print(f"--> Best Baseline Saved! (Dice: {best_dice:.4f})")
            # Save training history alongside best model
            history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Training history saved to {history_path}")

    # Save final training history
    history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Final training history saved to {history_path}")
    print("Baseline Training Complete.")


def train_distillation(config, device):
    """Train student model with Knowledge Distillation from teacher."""
    print(f"--- Starting DISTILLATION Training on {device} ---")

    exp_name = config['experiment_name']
    os.makedirs(config['save_dir'], exist_ok=True)

    train_loader, val_loader = create_dataloaders(config)

    # Initialize models
    print("Initializing Models...")
    student = StudentModel(encoder_name=config['student_backbone']).to(device)
    teacher = Teacher(checkpoint_path=config['teacher_checkpoint'], device=device)

    loss_fn_sup = DiceBCELoss().to(device)
    loss_fn_kd = KDLoss(temperature=config['temperature']).to(device)

    optimizer = optim.AdamW(student.parameters(), lr=config['learning_rate'])

    best_dice = 0.0
    history = {
        'model_name': 'Distillation',
        'epochs': [],
        'train_loss': [],
        'val_dice': [],
        'val_iou': []
    }

    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch + 1}/{config['epochs']}]")

        train_loss = train_distillation_epoch(
            student, teacher, train_loader, optimizer, loss_fn_sup, loss_fn_kd, device, config
        )
        val_dice, val_iou = validate(student, val_loader, device)

        print(f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config['save_dir'], f"{exp_name}_best.pth")
            torch.save(student.state_dict(), save_path)
            print(f"--> Best Model Saved! (Dice: {best_dice:.4f})")
            # Save training history alongside best model
            history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Training history saved to {history_path}")

    # Save final training history
    history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Final training history saved to {history_path}")
    print("Distillation Training Complete.")


def train_teacher(config, device):
    """Train the teacher model from scratch."""
    print(f"--- Starting TEACHER Training on {device} ---")

    exp_name = f"{config['experiment_name']}_TEACHER"
    os.makedirs(config['save_dir'], exist_ok=True)

    train_loader, val_loader = create_dataloaders(config)

    # Initialize teacher model (no checkpoint loading)
    teacher = Teacher(device=device)

    loss_fn = DiceBCELoss().to(device)
    optimizer = optim.AdamW(teacher.parameters(), lr=config['learning_rate'])

    best_dice = 0.0
    history = {
        'model_name': 'Teacher',
        'epochs': [],
        'train_loss': [],
        'val_dice': [],
        'val_iou': []
    }

    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch + 1}/{config['epochs']}]")

        train_loss = train_baseline_epoch(teacher, train_loader, optimizer, loss_fn, device)
        val_dice, val_iou = validate(teacher, val_loader, device)

        print(f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config['save_dir'], f"{exp_name}_best.pth")
            torch.save(teacher.state_dict(), save_path)
            print(f"--> Best Teacher Saved! (Dice: {best_dice:.4f})")
            # Save training history alongside best model
            history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"Training history saved to {history_path}")

    # Save final training history
    history_path = os.path.join(config['save_dir'], f"{exp_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Final training history saved to {history_path}")
    print("Teacher Training Complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Script for Polyp Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  baseline      Train student model from scratch (no teacher)
  distillation  Train student with Knowledge Distillation from teacher
  teacher       Train the teacher model from scratch

Examples:
  python train.py --mode baseline --config configs/config.yaml
  python train.py --mode distillation --config configs/config.yaml
  python train.py --mode teacher --config configs/config.yaml
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'distillation', 'teacher'],
        default='distillation',
        help='Training mode (default: distillation)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'baseline':
        train_baseline(config, device)
    elif args.mode == 'distillation':
        train_distillation(config, device)
    elif args.mode == 'teacher':
        train_teacher(config, device)


if __name__ == "__main__":
    main()
