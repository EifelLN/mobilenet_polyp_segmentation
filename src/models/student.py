import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional

class StudentModel(nn.Module):
    """Lightweight U-Net with MobileNetV2 backbone for efficient inference."""

    def __init__(self, num_classes: int = 1, encoder_name: str = "mobilenet_v2"):
        super(StudentModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None  # Returns raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_parameter_count(self) -> int:
        """Returns number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Sanity Check
    model = StudentModel()
    print(f"Student Model (MobileNetV2) Loaded.")
    print(f"Total Parameters: {model.get_parameter_count():,}")
    
    # Test Forward Pass
    dummy_input = torch.randn(2, 3, 320, 320)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape} (Expected: 2, 1, 320, 320)")