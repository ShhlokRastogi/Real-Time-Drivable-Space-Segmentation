import os
import shutil
from tqdm import tqdm

def merge_datasets(source_dirs, out_dir):
    print(f"Creating merged dataset at {out_dir}...")
    
    img_dir_out = os.path.join(out_dir, 'images')
    mask_dir_out = os.path.join(out_dir, 'binary_masks')
    overlay_dir_out = os.path.join(out_dir, 'visual_overlays')
    
    os.makedirs(img_dir_out, exist_ok=True)
    os.makedirs(mask_dir_out, exist_ok=True)
    os.makedirs(overlay_dir_out, exist_ok=True)
    
    total_copied = 0
    
    for src in source_dirs:
        print(f"\nProcessing {src}...")
        img_dir_in = os.path.join(src, 'images')
        mask_dir_in = os.path.join(src, 'binary_masks')
        overlay_dir_in = os.path.join(src, 'visual_overlays')
        
        if not os.path.exists(img_dir_in):
            print(f"Directory {img_dir_in} does not exist, skipping.")
            continue
            
        images = [f for f in os.listdir(img_dir_in) if f.endswith(('.jpg', '.png'))]
        for img in tqdm(images, desc=f"Copying {os.path.basename(src)} files"):
            base_name = os.path.splitext(img)[0]
            
            # The files are already uniquely named by the generators (e.g. sample_00000 / train_idd_00000)
            # but to ensure absolutely 0 collision we could prefix them, but they are already unique.
            # We'll just copy them safely.
            
            src_img = os.path.join(img_dir_in, img)
            dst_img = os.path.join(img_dir_out, img)
            
            # Determine mask extension. Usually .png
            src_mask = os.path.join(mask_dir_in, f"{base_name}.png")
            dst_mask = os.path.join(mask_dir_out, f"{base_name}.png")
            
            # Overlay is usually _overlay.jpg
            src_overlay = os.path.join(overlay_dir_in, f"{base_name}_overlay.jpg")
            dst_overlay = os.path.join(overlay_dir_out, f"{base_name}_overlay.jpg")
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            if os.path.exists(src_overlay):
                shutil.copy2(src_overlay, dst_overlay)
                
            total_copied += 1
            
    print(f"\nSuccessfully compiled '{out_dir}' with {total_copied} total samples!")

if __name__ == "__main__":
    sources = [
        'c:/drivableseg/exported_data_idd20k',
        'c:/drivableseg/exported_data_perfected'
    ]
    target = 'c:/drivableseg/dataset_id10k_nuscenes'
    merge_datasets(sources, target)
