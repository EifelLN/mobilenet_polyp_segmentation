# ============================================================================
# POLYP SEGMENTATION - EVALUATION (GOOGLE COLAB VERSION)
# ============================================================================
# Single-file version of evaluate.py for Google Colab.
# Evaluates trained models on multiple test datasets.
# ============================================================================

# =============================================================================
# CELL 1: INSTALL DEPENDENCIES
# =============================================================================
# !pip install segmentation-models-pytorch albumentations tqdm opencv-python

# =============================================================================
# CELL 2: IMPORTS
# =============================================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import glob
import json
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import segmentation_models_pytorch as smp

# =============================================================================
# CELL 3: CONFIGURATION
# =============================================================================
# Edit these settings for your evaluation

CONFIG = {
    # Paths (Adjust for Colab)
    "data_root": "dataset",
    "results_dir": "results",
    
    # Model Settings
    "img_size": 320,
    
    # Checkpoints to evaluate (add your trained models here)
    "checkpoints": {
        "Baseline": "checkpoints/MobileNetV2_Baseline_v1_best.pth",
        "Teacher": "checkpoints/Teacher_ResNet50_v1_best.pth",
        "Distillation": "checkpoints/MobileNetV2_Distillation_v1_best.pth",
    },
    
    # Set to True to save visual predictions
    "save_visuals": True,
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
    
    return {
        "dice": dice.item(),
        "iou": iou.item()
    }


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
# CELL 5: DATA LOADER
# =============================================================================

class PolypDataset(Dataset):
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

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0

        return image, mask

# =============================================================================
# CELL 6: MODEL DEFINITIONS
# =============================================================================

class StudentModel(nn.Module):
    """Lightweight U-Net with MobileNetV2 backbone."""
    def __init__(self, num_classes: int = 1, encoder_name: str = "mobilenet_v2"):
        super(StudentModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


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
# CELL 7: TRANSFORMS
# =============================================================================

def get_test_transform(img_size):
    """Test-time transforms: resize only."""
    return A.Compose([
        A.Resize(img_size, img_size),
    ])

# =============================================================================
# CELL 8: VISUALIZATION
# =============================================================================

def save_prediction(image_tensor, mask_tensor, pred_logits, save_path):
    """Save visual comparison: Image | Ground Truth | Prediction."""
    # Un-normalize image
    image = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process Masks
    gt = mask_tensor.squeeze().cpu().numpy() * 255.0
    gt = gt.astype(np.uint8)
    
    pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
    pred_binary = (pred_prob > 0.5).astype(np.uint8) * 255
    
    # Make masks 3-channel for concatenation
    gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([image, gt_color, pred_color])
    cv2.imwrite(save_path, combined)

# =============================================================================
# CELL 9: EVALUATION FUNCTION
# =============================================================================

def evaluate(model, loader, device, save_dir=None):
    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    time_meter = AverageMeter()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Measure Inference Time
            start = time.time()
            logits = model(images)
            end = time.time()
            
            batch_time = end - start
            time_meter.update(batch_time)
            
            # Metrics
            metrics = calculate_metrics(logits, masks)
            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))
            
            # Save first 10 samples for visualization
            if save_dir and i < 10: 
                save_path = os.path.join(save_dir, f"sample_{i}.png")
                save_prediction(images[0], masks[0], logits[0], save_path)
                
    fps = 1.0 / time_meter.avg if time_meter.avg > 0 else 0
    return dice_meter.avg, iou_meter.avg, fps

# =============================================================================
# CELL 10: MAIN EVALUATION SCRIPT
# =============================================================================

def evaluate_model(model_name, checkpoint_path, model_type="student"):
    """
    Evaluate a single model on all test datasets.
    
    Args:
        model_name: Name for display/saving
        checkpoint_path: Path to .pth file
        model_type: "student" for MobileNetV2, "teacher" for ResNet50
    """
    config = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load Model
    if model_type == "teacher":
        model = HeavyTeacher()
    else:
        model = StudentModel()
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model Parameters: {model.get_parameter_count():,}")
    
    # Find All Test Datasets
    test_root = os.path.join(config['data_root'], 'TestDataset')
    test_folders = sorted(glob.glob(os.path.join(test_root, "*")))
    
    if not test_folders:
        print(f"No test folders found in {test_root}")
        return None

    print(f"\nFound {len(test_folders)} test datasets")
    print("-" * 60)
    print(f"{'Dataset':<20} | {'Dice':<10} | {'IoU':<10} | {'FPS':<10}")
    print("-" * 60)

    # Store results
    results = {
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "datasets": {}
    }
    
    total_dice = 0
    total_iou = 0
    
    for folder_path in test_folders:
        dataset_name = os.path.basename(folder_path)
        
        if not os.path.exists(os.path.join(folder_path, "images")):
            continue

        dataset = PolypDataset(
            root_dir=folder_path,
            transform=get_test_transform(config['img_size']),
            img_size=config['img_size']
        )
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        # Visuals Path
        save_vis_path = None
        if config['save_visuals']:
            save_vis_path = os.path.join(config['results_dir'], "visuals", model_name, dataset_name)
        
        dice, iou, fps = evaluate(model, loader, device, save_dir=save_vis_path)
        
        print(f"{dataset_name:<20} | {dice:.4f}     | {iou:.4f}     | {fps:.1f}")
        
        results["datasets"][dataset_name] = {
            "dice": dice,
            "iou": iou,
            "fps": fps
        }
        
        total_dice += dice
        total_iou += iou
        
    n_datasets = len([f for f in test_folders if os.path.exists(os.path.join(f, "images"))])
    avg_dice = total_dice / n_datasets if n_datasets > 0 else 0
    avg_iou = total_iou / n_datasets if n_datasets > 0 else 0
    
    print("-" * 60)
    print(f"{'AVERAGE':<20} | {avg_dice:.4f}     | {avg_iou:.4f}")
    
    results["average_dice"] = avg_dice
    results["average_iou"] = avg_iou
    
    return results


def main():
    config = CONFIG
    
    print("=" * 60)
    print("POLYP SEGMENTATION - MODEL EVALUATION")
    print("=" * 60)
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    all_results = {}
    
    # Evaluate each configured checkpoint
    for model_name, checkpoint_path in config['checkpoints'].items():
        # Determine model type based on name
        if "Teacher" in model_name or "teacher" in model_name.lower():
            model_type = "teacher"
        else:
            model_type = "student"
        
        results = evaluate_model(model_name, checkpoint_path, model_type)
        
        if results:
            all_results[model_name] = results
    
    # Save all results to JSON
    if all_results:
        results_path = os.path.join(config['results_dir'], "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n\nAll results saved to: {results_path}")
    
    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Model':<20} | {'Avg Dice':<12} | {'Avg IoU':<12}")
        print("-" * 70)
        for model_name, results in all_results.items():
            print(f"{model_name:<20} | {results['average_dice']*100:.2f}%        | {results['average_iou']*100:.2f}%")
        print("=" * 70)


# =============================================================================
# CELL 11: RUN EVALUATION
# =============================================================================
# Uncomment to run evaluation
# main()

# Or evaluate a single model:
# results = evaluate_model("My_Model", "checkpoints/my_model_best.pth", model_type="student")

if __name__ == "__main__":
    main()
