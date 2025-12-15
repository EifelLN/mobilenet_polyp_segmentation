import os
import argparse
import yaml
import glob
import time
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Project Imports
from models.student import StudentModel
from models.teacher import Teacher
from utils.data_loader import PolypDataset
from utils.metrics import calculate_metrics, AverageMeter

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_test_transform(img_size):
    """
    Test-time transforms: Only Resize, no augmentations.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
    ])

def save_prediction(image_tensor, mask_tensor, pred_logits, save_path):
    """Saves a visual comparison: Image | Ground Truth | Prediction."""
    # Un-normalize image for visualization
    image = image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process masks
    gt = mask_tensor.squeeze().cpu().numpy() * 255.0
    gt = gt.astype(np.uint8)
    
    pred_prob = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
    pred_binary = (pred_prob > 0.5).astype(np.uint8) * 255
    
    # Concatenate side-by-side (convert masks to 3-channel for concatenation)
    gt_color = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor(pred_binary, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([image, gt_color, pred_color])
    
    cv2.imwrite(save_path, combined)

def evaluate(model, loader, device, save_dir=None):
    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    # Create directory for visuals if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Warmup runs to avoid CUDA kernel compilation overhead
    warmup_iterations = 10
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= warmup_iterations:
                break
            images = images.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Actual evaluation with proper timing
    total_inference_time = 0.0
    total_images = 0
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Synchronize before timing for accurate GPU measurement
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            logits = model(images)
            
            # Synchronize after inference to ensure GPU work is complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            batch_time = time.perf_counter() - start
            total_inference_time += batch_time
            total_images += images.size(0)
            
            metrics = calculate_metrics(logits, masks)
            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))
            
            if save_dir and i < 10: 
                save_path = os.path.join(save_dir, f"sample_{i}.png")
                save_prediction(images[0], masks[0], logits[0], save_path)
                
    fps = total_images / total_inference_time if total_inference_time > 0 else 0
    return dice_meter.avg, iou_meter.avg, fps

def main():
    parser = argparse.ArgumentParser(description="Evaluate polyp segmentation models")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained .pth file')
    parser.add_argument('--save_visuals', action='store_true', help='Save predicted masks')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluation Started on {device} ---")
    print(f"Loading Checkpoint: {args.checkpoint}")
    
    # Load Model
    model = StudentModel(encoder_name=config['student_backbone'])
    
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    model.to(device)
    model.eval()
    
    test_root = os.path.join(config['data_root'], 'TestDataset')
    test_folders = sorted(glob.glob(os.path.join(test_root, "*")))
    
    if not test_folders:
        print(f"No test folders found in {test_root}")
        return

    print(f"\nFound {len(test_folders)} test datasets: {[os.path.basename(f) for f in test_folders]}")
    print("-" * 60)
    print(f"{'Dataset':<20} | {'Dice':<10} | {'IoU':<10} | {'FPS':<10}")
    print("-" * 60)

    total_dice = 0
    
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
        save_vis_path = os.path.join("results", "visuals", dataset_name) if args.save_visuals else None
        
        dice, iou, fps = evaluate(model, loader, device, save_dir=save_vis_path)
        
        print(f"{dataset_name:<20} | {dice:.4f}     | {iou:.4f}     | {fps:.1f}")
        total_dice += dice
        
    print("-" * 60)
    print(f"Average Dice across all datasets: {total_dice / len(test_folders):.4f}")

if __name__ == "__main__":
    main()