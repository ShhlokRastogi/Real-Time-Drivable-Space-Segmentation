import os
import json
import cv2
import numpy as np
import glob
from tqdm import tqdm

def export_idd20k_dataset(
    idd_root='c:/drivableseg/id20k/idd20kII', 
    out_dir='c:/drivableseg/exported_data_idd20k', 
    target_size=(640, 360),
    split='train',
    max_samples=None
):
    """
    Parses IDD20k polygons and exports to the offline dataset format:
    resizes images to 640x360, creates geometric perfect binary masks for drivable area,
    and visual overlays for quick verification.
    """
    img_dir_out = os.path.join(out_dir, 'images')
    mask_dir_out = os.path.join(out_dir, 'binary_masks')
    overlay_dir_out = os.path.join(out_dir, 'visual_overlays')
    
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)
    os.makedirs(overlay_dir_out, exist_ok=True)
    
    # Drivable area classes
    drivable_classes = {"road", "drivable fallback"}
    
    # Discover all JSON polygon files
    print(f"Scanning {split} split for JSON polygon definitions...")
    json_search_path = os.path.join(idd_root, 'gtFine', split, '**', '*_gtFine_polygons.json')
    json_files = glob.glob(json_search_path, recursive=True)
    
    if max_samples is not None:
        json_files = json_files[:max_samples]
        
    total_samples = len(json_files)
    print(f"Found {total_samples} samples. Beginning geometric rendering to {out_dir}/ ...")
    
    success_count = 0
    
    for i, json_path in enumerate(tqdm(json_files)):
        # Determine paths
        # IDD file structure: 
        # gtFine/train/201/frameXXXX_gtFine_polygons.json
        # leftImg8bit/train/201/frameXXXX_leftImg8bit.jpg (or .png)
        
        base_name = os.path.basename(json_path).replace('_gtFine_polygons.json', '')
        parent_dir = os.path.basename(os.path.dirname(json_path))
        
        # Searching for the image file
        img_path = os.path.join(idd_root, 'leftImg8bit', split, parent_dir, f"{base_name}_leftImg8bit.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(idd_root, 'leftImg8bit', split, parent_dir, f"{base_name}_leftImg8bit.png")
            if not os.path.exists(img_path):
                print(f"Warning: Image not found for {base_name}, skipping.")
                continue
                
        # 1. Load Original Image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 2. Parse JSON and Draw Mask at original resolution
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            continue
            
        mask_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # In IDD/Cityscapes, objects can overlap. 'road' often spans under foreground objects.
        # We must draw drivable areas first, then draw all Non-drivable areas as 0 to cut them out.
        
        # Step 1: Draw Drivable Areas (value 1 and 2)
        for obj in data.get('objects', []):
            label = obj.get('label', '')
            deleted = obj.get('deleted', 0)
            if deleted: continue
            
            if label == 'road':
                polygon = obj.get('polygon', [])
                if not polygon: continue
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_raw, [pts], 1)
            elif label == 'drivable fallback':
                polygon = obj.get('polygon', [])
                if not polygon: continue
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_raw, [pts], 2)
                
        # Step 2: Cut out Obstacles/Non-drivable areas (value 0)
        # Any object that is NOT in drivable_classes will be subtracted from the mask
        for obj in data.get('objects', []):
            label = obj.get('label', '')
            deleted = obj.get('deleted', 0)
            if deleted: continue
            
            if label not in drivable_classes:
                polygon = obj.get('polygon', [])
                if not polygon: continue
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_raw, [pts], 0)
        
        # 3. Resize precisely to Network Input Dimensions
        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_raw, target_size, interpolation=cv2.INTER_NEAREST)
        
        # 4. Generate Visual Overlay (Green for Road, Blue for Fallback)
        colored_mask = np.zeros_like(img_resized)
        colored_mask[mask_resized == 1] = [0, 255, 0]  # Green overlay (BGR)
        colored_mask[mask_resized == 2] = [255, 0, 0]  # Blue overlay (BGR)
        alpha = 0.45
        blended_output = cv2.addWeighted(colored_mask, alpha, img_resized, 1 - alpha, 0)
        
        # 5. Save outputs in universal sequence naming for DataLoader
        filename = f"{split}_idd_{i:05d}"
        
        cv2.imwrite(os.path.join(img_dir_out, f"{filename}.jpg"), img_resized)
        cv2.imwrite(os.path.join(mask_dir_out, f"{filename}.png"), mask_resized)
        cv2.imwrite(os.path.join(overlay_dir_out, f"{filename}_overlay.jpg"), blended_output)
        
        success_count += 1
        
    print(f"\nIDD20k Dataset successfully exported! ({success_count} samples processed)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Process only 50 images to test the pipeline.")
    args = parser.parse_args()
    
    max_samples = 50 if args.test_run else None
    
    # For a full export you could call it on 'train' and 'val'
    export_idd20k_dataset(split='train', max_samples=max_samples)
    if not args.test_run:
        export_idd20k_dataset(split='val', max_samples=None)
