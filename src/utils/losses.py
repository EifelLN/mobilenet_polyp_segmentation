import torch
import torch.nn as nn
import torch.nn.functional as F

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


class KDLoss(nn.Module):
    """Knowledge Distillation loss using soft BCE targets."""
    
    def __init__(self, temperature=4.0):
        super(KDLoss, self).__init__()
        self.T = temperature
        self.soft_target_loss = nn.BCEWithLogitsLoss()

    def forward(self, student_logits, teacher_logits):
        if student_logits.shape != teacher_logits.shape:
            teacher_logits = F.interpolate(
                teacher_logits, 
                size=student_logits.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        soft_student = student_logits / self.T
        soft_teacher = teacher_logits / self.T
        teacher_probs = torch.sigmoid(soft_teacher).detach()
        loss = self.soft_target_loss(soft_student, teacher_probs) * (self.T ** 2)
        
        return loss

if __name__ == "__main__":
    # Sanity Check
    print("Testing Loss Functions...")
    
    # Dummy data
    B, C, H, W = 2, 1, 320, 320
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    teacher_pred = torch.randn(B, C, 352, 352) # Simulate resolution mismatch

    # Test DiceBCE
    criterion_supervised = DiceBCELoss()
    loss_sup = criterion_supervised(pred, target)
    print(f"Supervised Loss (Dice+BCE): {loss_sup.item():.4f}")

    # Test KD
    criterion_kd = KDLoss(temperature=4)
    loss_kd = criterion_kd(pred, teacher_pred)
    print(f"Distillation Loss (T=4): {loss_kd.item():.4f}")
    
    # Check resizing logic
    assert loss_kd.item() > 0, "KD Loss should be non-zero"
    print("Test passed.")