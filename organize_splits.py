import os
import shutil
from tqdm import tqdm

def restructure_dataset(base_dir='c:/drivableseg/dataset_id10k_nuscenes'):
    print(f"Restructuring dataset at {base_dir} into splits...")
    
    # Existing subdirectories
    img_dir_in = os.path.join(base_dir, 'images')
    mask_dir_in = os.path.join(base_dir, 'binary_masks')
    overlay_dir_in = os.path.join(base_dir, 'visual_overlays')
    
    if not os.path.exists(img_dir_in):
        print("Data not found. Did you already reorganize it?")
        return
        
    # We will create train/ val/ test/ inside base_dir, and inside each, images/ binary_masks/ visual_overlays/
    splits = ['train', 'val', 'test']
    
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'binary_masks'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'visual_overlays'), exist_ok=True)
        
    images = [f for f in os.listdir(img_dir_in) if f.endswith(('.jpg', '.png'))]
    
    moved_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for img in tqdm(images, desc="Organizing files by split"):
        # Determine split from filename prefix
        target_split = 'train' # default fallback
        
        if img.startswith('train_'):
            target_split = 'train'
        elif img.startswith('val_'):
            target_split = 'val'
        elif img.startswith('test_'):
            target_split = 'test'
        elif img.startswith('sample_'):
            # nuScenes perfect dataset doesn't explicitly denote split in name
            # By convention, we'll put it in train.
            target_split = 'train'
            
        base_name = os.path.splitext(img)[0]
        
        # Paths to original files
        src_img = os.path.join(img_dir_in, img)
        src_mask = os.path.join(mask_dir_in, f"{base_name}.png")
        src_overlay = os.path.join(overlay_dir_in, f"{base_name}_overlay.jpg")
        
        # Target paths
        dst_img = os.path.join(base_dir, target_split, 'images', img)
        dst_mask = os.path.join(base_dir, target_split, 'binary_masks', f"{base_name}.png")
        dst_overlay = os.path.join(base_dir, target_split, 'visual_overlays', f"{base_name}_overlay.jpg")
        
        # Move files
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        if os.path.exists(src_mask):
            shutil.move(src_mask, dst_mask)
        if os.path.exists(src_overlay):
            shutil.move(src_overlay, dst_overlay)
            
        moved_counts[target_split] += 1
        
    print("\nCleanup: Removing old flat directories if empty...")
    try:
        os.rmdir(img_dir_in)
        os.rmdir(mask_dir_in)
        os.rmdir(overlay_dir_in)
        print("Cleanup successful.")
    except Exception as e:
        print(f"Cleanup note: {e}")
        
    print("\nRestructuring complete!")
    print(f"Stats: Train={moved_counts['train']}, Val={moved_counts['val']}, Test={moved_counts['test']}")

if __name__ == "__main__":
    restructure_dataset()
