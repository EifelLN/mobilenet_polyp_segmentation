import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Teacher(nn.Module):
    """U-Net++ with ResNet-50 encoder for knowledge distillation."""
    
    def __init__(self, num_classes=1, checkpoint_path=None, device='cuda'):
        super(Teacher, self).__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet50",        
            encoder_weights="imagenet",     
            in_channels=3,
            classes=num_classes,
            activation=None                 
        )
        
        self.to(device)
        
        if checkpoint_path:
            self.load_weights(checkpoint_path)
        
    def load_weights(self, path):
        """Load pretrained weights, handling 'model.' prefix if present."""
        print(f"Loading Teacher weights from {path}...")
        state_dict = torch.load(path, map_location=next(self.model.parameters()).device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k.replace('model.', '', 1)
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.eval()
        print("Teacher Ready.")

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Sanity Check
    try:
        model = Teacher(device='cpu')
        print(f"Heavy Teacher Loaded.")
    except Exception as e:
        print(f"Error: {e}")