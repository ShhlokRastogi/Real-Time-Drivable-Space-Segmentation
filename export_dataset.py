import os
import cv2
import numpy as np
from dataset_example import NuScenesDrivableDataset

def export_offline_dataset(dataroot='c:/drivableseg/v1.0-mini', out_dir='c:/drivableseg/exported_data_perfected'):
    """
    Pre-generates and saves all images, binary masks, AND visual overlays to disk.
    """
    img_dir = os.path.join(out_dir, 'images')
    mask_dir = os.path.join(out_dir, 'binary_masks')
    overlay_dir = os.path.join(out_dir, 'visual_overlays')
    
    # Create the offline destination folders
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    print("Loading NuScenes API for Offline Dataset Export...")
    dataset = NuScenesDrivableDataset(dataroot=dataroot, target_size=(640, 360))
    total_samples = len(dataset)
    
    print(f"Exporting exactly {total_samples} samples into {out_dir}/ ...")
    print("This might take a minute...")
    
    for i in range(total_samples):
        img_tensor, mask_tensor = dataset[i]
        
        # 1. Image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 2. Binary Mask
        mask_raw = mask_tensor.squeeze(0).numpy()
        mask_255 = (mask_raw * 255.0).astype(np.uint8) 
        
        # 3. Visual Overlay
        colored_mask = np.zeros_like(img_bgr)
        colored_mask[mask_raw > 0] = [0, 255, 0]  # Green overlay
        alpha = 0.45
        blended_output = cv2.addWeighted(colored_mask, alpha, img_bgr, 1 - alpha, 0)
        
        # 4. Save!
        filename = f"sample_{i:05d}"
        
        cv2.imwrite(os.path.join(img_dir, f"{filename}.jpg"), img_bgr)
        cv2.imwrite(os.path.join(mask_dir, f"{filename}.png"), mask_255)
        cv2.imwrite(os.path.join(overlay_dir, f"{filename}_overlay.jpg"), blended_output)
        
        if (i + 1) % 50 == 0 or (i + 1) == total_samples:
            print(f"[{i+1}/{total_samples}] Successfully exported.")
            
    print("\nDataset successfully and completely exported with visual overlays!")

if __name__ == "__main__":
    export_offline_dataset()
