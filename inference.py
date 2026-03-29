import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from model import RealTimeDeepLabV3Plus

def calculate_iou(pred, target):
    """
    Intersection Over Union (IoU) math.
    Strictly penalizes False Positives (bleeding over sidewalks) 
    and False Negatives (missing the road).
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0  
    return intersection / union

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Firing up Real-Time Inference Engine on {device} ---")
    
    # 1. Load the Custom Trained Artificial Brain
    model_path = "c:/drivableseg/drivable_model_latest.pth"
    if not os.path.exists(model_path):
        print(f"Error: Could not find checkpoint at {model_path}. Did train.py fail?")
        return
        
    print("Loading MobileNetV2-DeepLabV3Plus Topology into VRAM...")
    model = RealTimeDeepLabV3Plus(num_classes=1).to(device)
    
    # Load strictly to the available hardware to prevent mismatches
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Freeze dropout & batchnorm layers
    
    # 2. Setup Inputs / Outputs
    img_dir = "c:/drivableseg/dataset_id10k_nuscenes/val/images"
    mask_dir = "c:/drivableseg/dataset_id10k_nuscenes/val/binary_masks"
    out_dir = "c:/drivableseg/inference_results"
    os.makedirs(out_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    # Benchmark subset (First 50 images to get a highly accurate CPU/GPU FPS average)
    subset = img_files[:50]
    
    total_time = 0.0
    total_iou = 0.0
    
    print(f"Running automated benchmarking suite on {len(subset)} dashcam frames...")
    
    with torch.no_grad(): # Disable gradients completely for massive speed boost
        for i, img_name in enumerate(subset):
            img_path = os.path.join(img_dir, img_name)
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            # Load raw 640x360 image
            orig_img_cv = cv2.imread(img_path)
            
            # Prepare PyTorch constraints
            pil_img = Image.open(img_path).convert('RGB')
            img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # ==============================================================
            # START FPS CLOCK (Benchmarking Model Matrix Execution bounds)
            # ==============================================================
            start_time = time.time()
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    raw_pred = model(img_tensor)
            else:
                raw_pred = model(img_tensor) 
            
            # Collapse the output Logits into a clean [0, 1] Binary Threshold
            probs = torch.sigmoid(raw_pred).squeeze().cpu().numpy()
            binary_pred = (probs > 0.5).astype(np.uint8) 
            
            end_time = time.time()
            # ==============================================================
            # END FPS CLOCK
            # ==============================================================
            
            frame_time = end_time - start_time
            total_time += frame_time
            
            # Calculate Ground Truth IoU Accuracy dynamically
            iou = 0.0
            if os.path.exists(mask_path):
                # We benchmark against the perfectly-masked geometry data we exported earlier!
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_binary = (gt_mask > 127).astype(np.uint8)
                iou = calculate_iou(binary_pred, gt_binary)
                total_iou += iou
            
            # Generate the Human-QA Visualization Output 
            # We paint the AI's "brain prediction" in Bright Blue to distinguish it from ground truth!
            color_mask = np.zeros_like(orig_img_cv)
            color_mask[binary_pred == 1] = [255, 100, 0] # BGR Blue/Cyan hue
            
            # Merge 60% Dashcam + 40% AI Prediction
            overlay = cv2.addWeighted(orig_img_cv, 1.0, color_mask, 0.4, 0)
            
            # Watermark the active metrics onto the glass
            fps_val = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(overlay, f"FPS: {fps_val:.1f} | ACCURACY: {iou*100:.1f}%", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            # Save the execution
            out_path = os.path.join(out_dir, f"AI_pred_{img_name}")
            cv2.imwrite(out_path, overlay)
            
            if (i+1) % 10 == 0:
                print(f"Propagated {i+1}/{len(subset)} AI frames...")

    # Calculate global testing parameters for the Problem Statement Requirements
    avg_fps = len(subset) / total_time
    avg_iou = total_iou / len(subset)
    
    print("\n================================================")
    print("[RESULTS] AI INFERENCE BENCHMARK [RESULTS]")
    print("================================================")
    print(f"Target Hardware Configuration: {device.type.upper()}")
    print(f"Average Inference Speed:       {avg_fps:.2f} Frames/Second")
    print(f"Mean IoU Accuracy Score:       {avg_iou * 100:.2f}%")
    print("================================================")
    print(f"All inference diagnostic videos safely exported to: {out_dir}")

if __name__ == "__main__":
    run_inference()
