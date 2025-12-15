# ============================================================================
# POLYP SEGMENTATION - TEACHER TRAINING (GOOGLE COLAB VERSION)
# ============================================================================
# Single-file version of train_teacher_supervised.py for Google Colab.
# Trains Teacher model (ResNet50 U-Net++) with ground truth supervision.
# ============================================================================

# =============================================================================
# CELL 1: INSTALL DEPENDENCIES
# =============================================================================
# !pip install segmentation-models-pytorch albumentations tqdm pyyaml opencv-python

# =============================================================================
# CELL 2: IMPORTS
# =============================================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import glob
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp

# =============================================================================
# CELL 3: CONFIGURATION
# =============================================================================
# Edit these settings as needed for your training

CONFIG = {
    # Experiment Name (Used for saving checkpoints)
    "experiment_name": "Teacher_ResNet50_v1",
    
    # Paths (Adjust for Colab)
    "data_root": "dataset",  # Change to your dataset path in Colab
    "save_dir": "checkpoints",
    
    # Model Hyperparameters
    "teacher_backbone": "resnet50",
    
    # Training Hyperparameters
    "img_size": 320,          # 320x320 is a sweet spot
    "batch_size": 8,          # Lower to 4 if OOM occurs (Teacher is heavier)
    "epochs": 50,
    "learning_rate": 1e-4,
    "num_workers": 2,         # Colab works well with 2
    "seed": 42,
}

# =============================================================================
# CELL 4: UTILITY CLASSES
# =============================================================================

def calculate_metrics(pred_logits, target_mask, threshold=0.5):
    """Compute Dice and IoU scores for binary segmentation."""
    pred_probs = torch.sigmoid(pred_logits)
    pred_mask = (pred_probs > threshold).float()
    
    pred_flat = pred_mask.view(-1)
    target_flat = target_mask.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    total_area = pred_flat.sum() + target_flat.sum()
    union = total_area - intersection
    
    epsilon = 1e-6
    dice = (2.0 * intersection + epsilon) / (total_area + epsilon)
    iou = (intersection + epsilon) / (union + epsilon)
    
    return {"dice": dice.item(), "iou": iou.item()}


class AverageMeter:
    """Computes and stores running average values."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# =============================================================================
# CELL 5: LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice coefficient loss for binary segmentation."""
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice Loss + Binary Cross Entropy."""
    def __init__(self, weight=None, reduction='mean'):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='mean')
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return bce + dice

# =============================================================================
# CELL 6: DATA LOADER
# =============================================================================

class PolypDataset(Dataset):
    """PyTorch Dataset for polyp segmentation."""
    
    def __init__(self, root_dir, transform=None, img_size=320):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

        self.images_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
        self.masks_paths = sorted(glob.glob(os.path.join(root_dir, "masks", "*.png")))

        if len(self.images_paths) == 0:
            raise ValueError(f"No images found in {root_dir}/images")
        
        if len(self.images_paths) != len(self.masks_paths):
            raise ValueError(f"Mismatch: {len(self.images_paths)} images but {len(self.masks_paths)} masks")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]

        # Load with OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        # Tensor Conversion & Normalization
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        # Strict Binarization
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        return image, mask

# =============================================================================
# CELL 7: MODEL DEFINITION
# =============================================================================

class HeavyTeacher(nn.Module):
    """U-Net++ with ResNet-50 encoder."""
    
    def __init__(self, num_classes=1, encoder_name="resnet50"):
        super(HeavyTeacher, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,        
            encoder_weights="imagenet",     
            in_channels=3,
            classes=num_classes,
            activation=None                 
        )

    def forward(self, x):
        return self.model(x)
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# =============================================================================
# CELL 8: TRANSFORMS
# =============================================================================

def get_transforms(img_size, mode="train"):
    """Returns Albumentations transforms."""
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

# =============================================================================
# CELL 9: TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(teacher, loader, optimizer, loss_fn, device):
    teacher.train()
    losses = AverageMeter()
    
    pbar = tqdm(loader, desc="Training (Teacher)", leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        logits = teacher(images)

        if torch.isnan(logits).any():
            continue

        loss = loss_fn(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer.step()
        
        if not torch.isnan(loss):
            losses.update(loss.item(), images.size(0))
            pbar.set_postfix({"Loss": f"{losses.avg:.4f}"})
        
    return losses.avg


def validate(teacher, loader, device):
    teacher.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = teacher(images)
            metrics = calculate_metrics(outputs, masks)
            
            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))
            
    return dice_meter.avg, iou_meter.avg

# =============================================================================
# CELL 10: MAIN TRAINING SCRIPT
# =============================================================================

def main():
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting TEACHER Training on {device} ---")
    
    # Create Save Directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Data Loaders
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

    # Model
    print("Initializing Teacher Model (ResNet50 U-Net++)...")
    teacher = HeavyTeacher(encoder_name=config['teacher_backbone']).to(device)
    print(f"Teacher Parameters: {teacher.get_parameter_count():,}")

    # Loss & Optimizer
    loss_fn = DiceBCELoss().to(device)
    optimizer = optim.AdamW(teacher.parameters(), lr=config['learning_rate'])

    # Training Loop
    best_dice = 0.0
    
    # Training History
    history = {
        'model_name': 'Teacher',
        'epochs': [],
        'train_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        
        # Train
        train_loss = train_one_epoch(teacher, train_loader, optimizer, loss_fn, device)
        
        val_dice, val_iou = validate(teacher, val_loader, device)
        
        print(f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(config['save_dir'], f"{config['experiment_name']}_best.pth")
            torch.save(teacher.state_dict(), save_path)
            print(f"--> Best Model Saved! (Dice: {best_dice:.4f})")
    
    history_path = os.path.join(config['save_dir'], f"{config['experiment_name']}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")
    print("Training Complete.")

# =============================================================================
# CELL 11: RUN TRAINING
# =============================================================================
# Uncomment the line below to run training
# main()

# Or run it directly:
if __name__ == "__main__":
    main()
