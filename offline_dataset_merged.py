import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MergedRoadDataset(Dataset):
    """
    Direct PyTorch loader for the Merged Road Dataset.
    Reads Native 640x360 images and binary masks [0, 1] securely from the drive.
    """
    def __init__(self, data_split_dir='c:/drivableseg/merged_road_dataset/train'):
        self.img_dir = os.path.join(data_split_dir, 'images')
        self.mask_dir = os.path.join(data_split_dir, 'masks')
        
        # Pull all images
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        
        # Pre-assign standard output path for the image tensor
        img_path = os.path.join(self.img_dir, img_name)
        
        # Safely map to the expected mask name, checking both common extensions
        base_name = img_name.rsplit('.', 1)[0]
        mask_path_png = os.path.join(self.mask_dir, base_name + '.png')
        mask_path_jpg = os.path.join(self.mask_dir, base_name + '.jpg')
        
        if os.path.exists(mask_path_png):
            mask_path = mask_path_png
        elif os.path.exists(mask_path_jpg):
            mask_path = mask_path_jpg
        else:
            raise FileNotFoundError(f"CRITICAL ERROR: Mask file completely missing for image '{img_name}'. Checked both .png and .jpg variants.")
            
        # 1. Load the pristine Image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # 2. Load the binary mask as 8-bit integers
        mask = Image.open(mask_path).convert('L') 
        
        # 3. Securely Resize BOTH Image and Mask to identical 640x360 dimensions for Batch Stacking
        # We use purely NEAREST neighbor for the mask to completely avoid generating fake fractional label targets
        image = image.resize((640, 360), resample=Image.BILINEAR)
        mask = mask.resize((640, 360), resample=Image.NEAREST)
        
        # 4. Convert Image to Tensor [C, H, W] inside normalized range 0.0 - 1.0
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # 4. Convert Mask to LongTensor [H, W] and mathematically guarantee it collapses to binary [0, 1]
        # Any value > 0 becomes 1 (Road). Background is 0.
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        mask_tensor = torch.clamp(mask_tensor, 0, 1) 
        
        # Return pristine geometric tensors (NO AUGMENTATION APPLED)
        return image_tensor, mask_tensor
