import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class OfflineNuScenesDatasetAugmented(Dataset):
    """
    Extremely deep-augmented PyTorch loader that rigidly synchronizes Image and Multi-Class Mask
    spatial warping matrices together to force neural generalization without losing 0/1/2 targets.
    """
    def __init__(self, data_dir='c:/drivableseg/exported_data_idd20k', augment=True):
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'binary_masks')
        self.augment = augment
        
        # We assume files are sequenced safely
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        # Map .jpg to .png safely
        mask_name = img_name.rsplit('.', 1)[0] + '.png'
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load Raw bytes 
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 8-bit Tri-State Map
        
        import numpy as np
        # PyTorch format: [Ch, H, W] for images
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        # PyTorch format: [H, W] categorical integers for CrossEntropy arrays
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        # -------------------------------------------------------------
        # AGGRESSIVE SPATIAL AUGMENTATION INJECTION
        # -------------------------------------------------------------
        if self.augment:
            # 1. Randomized Synchronized Affine Spatial Traversed Distortion Matrix
            if random.random() > 0.4:
                # Build random geometric matrix
                angle = random.uniform(-10.0, 10.0) # Simulate hilly terrain / sharp camera tilts
                translate = [random.uniform(-0.1, 0.1) * 640.0, random.uniform(-0.05, 0.1) * 360.0] # Push frame X/Y boundaries
                scale = random.uniform(0.85, 1.25) # Simulate zooming in or flying out
                shear = random.uniform(-5.0, 5.0) # Skew perspective mildly
                
                # Image can interpolate Bilinear
                image_tensor = TF.affine(
                    image_tensor, angle=angle, translate=translate, scale=scale, shear=[shear],
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                
                # MASK MUST BE STRICTLY NEAREST, and temporarily requires an unsqueezed dummy channel!
                mask_tensor = mask_tensor.unsqueeze(0).float()
                mask_tensor = TF.affine(
                    mask_tensor, angle=angle, translate=translate, scale=scale, shear=[shear],
                    interpolation=TF.InterpolationMode.NEAREST
                )
                mask_tensor = mask_tensor.squeeze(0).long()
                
            # 2. Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                # Ensure mask is flipped explicitly using TorchVision helper (which handles 2D tensors fine!)
                mask_tensor = mask_tensor.unsqueeze(0)
                mask_tensor = TF.hflip(mask_tensor).squeeze(0)
                
            # 3. Intense Native Color/Lighting Jitter (Image Only)
            if random.random() > 0.5:
                brightness = random.uniform(0.6, 1.4)
                contrast = random.uniform(0.6, 1.4)
                
                image_tensor = TF.adjust_brightness(image_tensor, brightness)
                image_tensor = TF.adjust_contrast(image_tensor, contrast)
                image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
        
        return image_tensor, mask_tensor
