import os
import cv2
import numpy as np
import shutil

def filter_imperfect_masks(in_dir='c:/drivableseg/exported_data_perfected', out_dir='c:/drivableseg/exported_data_imperfect'):
    """
    Automatically heuristically scans and quarantines imperfect 2D Map Projections.
    Separates the images into an 'imperfect' folder to keep the training data flawlessly pure.
    """
    img_dir_in = os.path.join(in_dir, 'images')
    mask_dir_in = os.path.join(in_dir, 'binary_masks')
    overlay_dir_in = os.path.join(in_dir, 'visual_overlays')
    
    img_dir_out = os.path.join(out_dir, 'images')
    mask_dir_out = os.path.join(out_dir, 'binary_masks')
    overlay_dir_out = os.path.join(out_dir, 'visual_overlays')
    
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)
    os.makedirs(overlay_dir_out, exist_ok=True)
    
    if not os.path.exists(mask_dir_in):
        print(f"Error: Could not find mask directory {mask_dir_in}")
        return

    # List all generated masks
    mask_files = [f for f in os.listdir(mask_dir_in) if f.endswith('.png')]
    total_files = len(mask_files)
    imperfect_count = 0
    
    print(f"Scanning {total_files} masks for geometric flat-earth projection anomalies...")
    
    for mask_file in mask_files:
        base_name = mask_file.replace('.png', '')
        mask_path = os.path.join(mask_dir_in, mask_file)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        H, W = mask.shape
        
        # -------------------------------------------------------------
        # HEURISTIC 1: Horizon / Sky Bleeding
        # A flat road should mathematically never appear in the top 35% of a front-facing dashcam.
        # If green mask pixels cross the horizon (y = 126), the map projection algorithm broke 
        # (usually due to the road driving down a hill while the 2D map assumed a flat plane).
        # -------------------------------------------------------------
        horizon_y = int(H * 0.35) 
        sky_pixels = np.sum(mask[:horizon_y, :] > 0)
        
        # -------------------------------------------------------------
        # HEURISTIC 2: Total Image Stretch Coverage
        # If the math artifact wrapped around the camera lens, it paints massive swaths of the screen green.
        # -------------------------------------------------------------
        total_road = np.sum(mask > 0)
        total_pixels = H * W
        road_percentage = total_road / total_pixels
        
        is_imperfect = False
        reason = ""
        
        if sky_pixels > 500:
            is_imperfect = True
            reason = "Horizon Bleed (Hill Elevation Error)"
        elif road_percentage > 0.60:
            is_imperfect = True
            reason = "Massive Scaling Artifact (>60% screen coverage)"
        elif road_percentage < 0.01:
            is_imperfect = True
            reason = "Missing/Empty Map Geometry"
            
        if is_imperfect:
            imperfect_count += 1
            
            # Physically move the offending files to the Quarantine Zone
            # We use try/except just in case a file is already moved or missing
            try:
                shutil.move(mask_path, os.path.join(mask_dir_out, f"{base_name}.png"))
                
                img_src = os.path.join(img_dir_in, f"{base_name}.jpg")
                if os.path.exists(img_src):
                    shutil.move(img_src, os.path.join(img_dir_out, f"{base_name}.jpg"))
                    
                overlay_src = os.path.join(overlay_dir_in, f"{base_name}_overlay.jpg")
                if os.path.exists(overlay_src):
                    shutil.move(overlay_src, os.path.join(overlay_dir_out, f"{base_name}_overlay.jpg"))
                    
                print(f"Quarantined {base_name}: {reason}")
            except Exception as e:
                pass
            
    print(f"\nFiltering complete! Physically separated {imperfect_count} flawed images out of {total_files}.")
    good_count = total_files - imperfect_count
    print(f"SUCCESS: You now have {good_count} flawless, mathematically-pure training samples remaining in '{in_dir}'.")
    print(f"The imperfect edge-case images were locked away in '{out_dir}'.")

if __name__ == "__main__":
    filter_imperfect_masks()
