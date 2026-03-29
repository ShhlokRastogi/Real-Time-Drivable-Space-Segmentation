import torch
import numpy as np
import cv2
import os
from dataset_example import NuScenesDrivableDataset
from torch.utils.data import DataLoader

def verify_dataset(nusc_version='v1.0-mini', dataroot='c:/drivableseg/v1.0-mini'):
    print("Initializing dataset for visual verification...")
    dataset = NuScenesDrivableDataset(nusc_version=nusc_version, dataroot=dataroot, target_size=(640, 360))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for safe quick test
    
    print("Fetching one random batch...")
    for images, masks in dataloader:
        print(f"Loaded batch. Image shape: {images.shape}, Mask shape: {masks.shape}")
        
        # We will loop through the batch (4 images)
        for i in range(images.size(0)):
            img_tensor = images[i]
            mask_tensor = masks[i]
            
            # 1. Convert PyTorch Image tensor [C, H, W] back to OpenCV-ready numpy array [H, W, C]
            img_np = img_tensor.permute(1, 2, 0).numpy() 
            img_np = (img_np * 255.0).astype(np.uint8)     # Denormalize [0, 1] to [0, 255]
            
            # 2. Convert from RGB (PIL format) to BGR (OpenCV format) so colors render properly
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 3. Handle the Binary Mask tensor [1, H, W] to [H, W]
            mask_np = mask_tensor.squeeze(0).numpy()
            
            # 4. Create an eye-catching visual overlay!
            # Let's paint the "Drivable Area" with a transparent Neon Green color.
            colored_mask = np.zeros_like(img_bgr)
            colored_mask[mask_np > 0] = [0, 255, 0]  # [B, G, R] formatting
            
            # Blend the original image and our green mask using cv2.addWeighted
            alpha = 0.45  # 45% transparency for the green tint
            blended_output = cv2.addWeighted(colored_mask, alpha, img_bgr, 1 - alpha, 0)
            
            # 5. Save the output to disk for you to review!
            output_filename = f"c:/drivableseg/verify_mask_{i}.jpg"
            cv2.imwrite(output_filename, blended_output)
            print(f"[{i+1}/4] Saved overlay visual to: {output_filename}")
            
        print("\nVerification complete! Open the generated .jpg files to inspect your data.")
        break # Exit after one batch
        
if __name__ == "__main__":
    verify_dataset()
