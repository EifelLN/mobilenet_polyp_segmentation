import torch
import numpy as np

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