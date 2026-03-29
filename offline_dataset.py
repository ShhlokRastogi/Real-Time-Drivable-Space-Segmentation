import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

class OfflineNuScenesDataset(Dataset):
    """
    Extremely high-speed PyTorch loader that reads our geometrically-perfected, 
    pre-rendered 640x360 dataset from the SSD in real-time.
    """
    def __init__(self, data_dir='c:/drivableseg/exported_data_perfected', augment=True):
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'binary_masks')
        self.augment = augment
        
        # We assume files are universally named sample_00000.jpg and sample_00000.png
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        mask_name = img_name.replace('.jpg', '.png') # Flawless pairing
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load Image and Mask natively 
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 8-bit grayscale
        
        import numpy as np
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        # -------------------------------------------------------------
        # NATIVE DATA AUGMENTATION (No External Dependencies!)
        # Any spatial distortion applied to the camera image MUST be 
        # mathematically identical on the semantic mask.
        # -------------------------------------------------------------
        if self.augment:
            # 1. Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                image_tensor = torch.flip(image_tensor, [2]) # Flip Width dimension ([C, H, W])
                mask_tensor = torch.flip(mask_tensor, [1])   # Flip Width dimension ([H, W])
                
            # 2. Native Brightness/Contrast Jitter (Image Only)
            if random.random() > 0.5:
                image_tensor = image_tensor * random.uniform(0.7, 1.3) # Brightness Math
                
                mean = torch.mean(image_tensor, dim=[1, 2], keepdim=True)
                image_tensor = (image_tensor - mean) * random.uniform(0.7, 1.3) + mean # Contrast Math
                
                image_tensor = torch.clamp(image_tensor, 0.0, 1.0) # Restrict bit bounds
        
        # (Removed binary bit boundary clamping for multi-class)
        
        return image_tensor, mask_tensor
