import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

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

# --- SANITY CHECK BLOCK ---
# This only runs if you execute 'python utils/data_loader.py' directly.
if __name__ == "__main__":
    import albumentations as A
    import matplotlib.pyplot as plt

    # Define a robust augmentation pipeline for training
    train_transform = A.Compose([
        A.Resize(320, 320),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        # RandomGamma simulates different lighting/monitor conditions
        A.RandomGamma(p=0.2), 
    ])

    print("Testing Data Loader...")
    
    # POINT THIS TO YOUR ACTUAL PATH
    # We use '..' to go up one level from utils/ to dataset/
    dataset_path = os.path.join("dataset", "TrainDataset") 
    
    try:
        ds = PolypDataset(dataset_path, transform=train_transform)
        print(f"Success! Found {len(ds)} samples.")
        
        # Fetch one sample
        img, mask = ds[10] # Pick index 10 just to see
        
        print(f"Image Tensor Shape: {img.shape}")
        print(f"Mask Tensor Shape: {mask.shape}")
        print(f"Max Value in Mask: {mask.max()} (Should be 1.0)")

        # Optional: Save a visual check
        # We have to undo the tensor conversion to plot it
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Augmented Image")
        plt.imshow(img.permute(1, 2, 0)) # (3,H,W) -> (H,W,3)
        
        plt.subplot(1, 2, 2)
        plt.title("Augmented Mask")
        plt.imshow(mask.squeeze(), cmap='gray') # (1,H,W) -> (H,W)
        
        plt.savefig("dataloader_check.png")
        print("Saved 'dataloader_check.png'. Check this file to verify alignment!")

    except Exception as e:
        print(f"\nError: {e}")