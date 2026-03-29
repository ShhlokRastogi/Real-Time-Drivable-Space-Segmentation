import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def export_bdd100k_dataset(
    bdd_root='c:/drivableseg/unprocessed_data', 
    out_dir='c:/drivableseg/exported_data_bdd100k', 
    target_size=(640, 360),
    split='train',
    max_samples=None
):
    """
    Parses the massive BDD100K JSON file and exports to the offline dataset format:
    resizes images to 640x360, creates geometric perfect binary masks for drivable area,
    and visual overlays for verification.
    """
    img_dir_out = os.path.join(out_dir, 'images')
    mask_dir_out = os.path.join(out_dir, 'binary_masks')
    overlay_dir_out = os.path.join(out_dir, 'visual_overlays')
    
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)
    os.makedirs(overlay_dir_out, exist_ok=True)
    
    # Path to the singular large JSON file
    json_path = os.path.join(bdd_root, split, 'annotations', f'bdd100k_labels_images_{split}.json')
    imgs_dir = os.path.join(bdd_root, split, 'images')
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return
        
    print(f"Loading mammoth BDD100K JSON File into memory ({json_path})...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return
        
    if max_samples is not None:
        data = data[:max_samples]
        
    total_samples = len(data)
    print(f"Found {total_samples} samples. Beginning geometric rendering to {out_dir}/ ...")
    
    success_count = 0
    
    for i, item in enumerate(tqdm(data)):
        img_name = item.get('name', '')
        if not img_name: continue
            
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        # 1. Load Original Image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 2. Parse JSON and Draw Mask at original resolution
        mask_raw = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # Step 1: Draw Drivable Areas (value 255)
        for label_obj in item.get('labels', []):
            category = label_obj.get('category', '')
            
            # Fill Drivable Polygons
            if category == 'drivable area':
                polys = label_obj.get('poly2d', [])
                for poly in polys:
                    vertices = poly.get('vertices', [])
                    if not vertices: continue
                    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_raw, [pts], 255)
                    
            # Draw Lane Lines as solid 12px structural bridges between disjoint areas
            elif category == 'lane':
                polys = label_obj.get('poly2d', [])
                for poly in polys:
                    vertices = poly.get('vertices', [])
                    if not vertices: continue
                    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(mask_raw, [pts], isClosed=False, color=255, thickness=16)
                    
        # Apply Morphological Closing to fuse tiny breaks or pixel-gaps between polygons
        kernel = np.ones((9, 9), np.uint8)
        mask_raw = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
        
        # 3. Resize precisely to Network Input Dimensions
        img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_raw, target_size, interpolation=cv2.INTER_NEAREST)
        
        # 4. Generate Visual Overlay (Neon Green + Original)
        colored_mask = np.zeros_like(img_resized)
        colored_mask[mask_resized > 0] = [0, 255, 0]  # Green overlay
        alpha = 0.45
        blended_output = cv2.addWeighted(colored_mask, alpha, img_resized, 1 - alpha, 0)
        
        # 5. Save outputs in universal sequence naming for DataLoader
        filename = f"{split}_bdd_{i:06d}"
        
        cv2.imwrite(os.path.join(img_dir_out, f"{filename}.jpg"), img_resized)
        cv2.imwrite(os.path.join(mask_dir_out, f"{filename}.png"), mask_resized)
        cv2.imwrite(os.path.join(overlay_dir_out, f"{filename}_overlay.jpg"), blended_output)
        
        success_count += 1
        
    print(f"\nBDD100K Dataset successfully exported! ({success_count} samples processed)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Process only 50 images to test the pipeline.")
    args = parser.parse_args()
    
    max_samples = 50 if args.test_run else None
    
    # Process train split
    export_bdd100k_dataset(split='train', max_samples=max_samples)
    
    # Process val split if full run
    if not args.test_run:
        export_bdd100k_dataset(split='val', max_samples=None)
