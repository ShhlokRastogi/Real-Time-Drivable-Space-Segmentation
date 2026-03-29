import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import argparse
from model import RealTimeDeepLabV3Plus
from loss import calculate_miou

def run_inference(input_dir, out_dir, model_path="c:/drivableseg/drivable_model_latest.pth", limit=None, mask_dir=None, prefix=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Firing up Real-Time Inference Pipeline on {device} ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find checkpoint at {model_path}. Did train.py fail?")
        return
        
    print("Loading Multi-Class MobileNetV2-UNet Topology into VRAM...")
    model = RealTimeDeepLabV3Plus(num_classes=3).to(device)
    
    # Load strictly to the available hardware to prevent mismatches
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Freeze dropout & batchnorm layers
    
    # [WARMUP PHASE]
    # PyTorch dynamically allocates CUDA memory and tunes cuDNN algorithms on the first few passes.
    # We must warm up the engine so the first frame doesn't skew our benchmarking.
    if device.type == 'cuda':
        print("Warming up CUDA cores to stabilize benchmarking...")
        dummy = torch.randn(1, 3, 360, 640).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize()
    
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.isfile(input_dir):
        img_files = [os.path.basename(input_dir)]
        input_dir = os.path.dirname(input_dir)
    else:
        img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    if prefix:
        img_files = [f for f in img_files if f.startswith(prefix)]
        
    if len(img_files) == 0:
        print(f"No valid images found in input directory: {input_dir}")
        return
        
    if limit is not None:
        img_files = img_files[:limit]
    
    total_time = 0.0
    total_miou = 0.0
    valid_masks = 0
    print(f"Running automated inference on {len(img_files)} frames from new data...")
    
    with torch.no_grad(): # Disable gradients completely for massive speed boost
        for i, img_name in enumerate(img_files):
            img_path = os.path.join(input_dir, img_name)
            
            # Load raw image
            orig_img_cv = cv2.imread(img_path)
            if orig_img_cv is None:
                continue
                
            orig_h, orig_w = orig_img_cv.shape[:2]
            
            # Resize image to 640x360 as required by the model architecture
            img_resized = cv2.resize(orig_img_cv, (640, 360))
            
            # Prepare PyTorch constraints
            pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # ==============================================================
            # START FPS CLOCK (Benchmarking Model Matrix Execution bounds)
            # ==============================================================
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    raw_pred = model(img_tensor)
                torch.cuda.synchronize() # MUST SYNCHRONIZE HERE to get true GPU runtime!
            else:
                raw_pred = model(img_tensor) 
            
            # Convert the raw logit outputs into absolute class predictions via Argmax
            class_pred = torch.argmax(raw_pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            end_time = time.time()
            # ==============================================================
            # END FPS CLOCK
            # ==============================================================
            
            frame_time = end_time - start_time
            total_time += frame_time
            
            # Resize pred back to original dimensions for overlaying
            class_pred_resized = cv2.resize(class_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Ground-Truth Metric Calculation against real data
            current_miou_str = ""
            if mask_dir is not None:
                mask_name = img_name.replace('.jpg', '.png')
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    # Load 8-bit Ground Truth mask
                    gt_mask_pil = Image.open(mask_path).convert('L')
                    gt_mask_np = np.array(gt_mask_pil)
                    
                    # Both class_pred (which is generated at 640x360) and gt_mask (from our exported dataset which is natively 640x360)
                    # can be directly compared. If shapes don't align perfectly, resize GT to match prediction matrix.
                    if gt_mask_np.shape != class_pred.shape:
                        gt_mask_np = cv2.resize(gt_mask_np, (class_pred.shape[1], class_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    calc_val = calculate_miou(torch.from_numpy(class_pred), torch.from_numpy(gt_mask_np), num_classes=3)
                    total_miou += calc_val
                    valid_masks += 1
                    current_miou_str = f" | mIoU: {calc_val:.2f}"
            
            # Generate the Human-QA Visualization Output 
            # 1: High Priority Road (Green), 2: Low Priority Fallback (Blue)
            color_mask = np.zeros_like(orig_img_cv)
            color_mask[class_pred_resized == 1] = [0, 255, 0] # BGR Green 
            color_mask[class_pred_resized == 2] = [255, 0, 0] # BGR Blue
            
            # Merge 60% Dashcam + 40% AI Prediction
            overlay = cv2.addWeighted(orig_img_cv, 1.0, color_mask, 0.4, 0)
            
            # Watermark the active metrics onto the glass
            fps_val = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(overlay, f"FPS: {fps_val:.1f}{current_miou_str}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            # Save the execution
            out_path = os.path.join(out_dir, f"AI_pred_{img_name}")
            cv2.imwrite(out_path, overlay)
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(img_files)} frames...")

    # Calculate global testing parameters
    avg_fps = len(img_files) / total_time
    
    print("\n================================================")
    print("[RESULTS] AI INFERENCE PIPELINE [RESULTS]")
    print("================================================")
    print(f"Target Hardware Configuration: {device.type.upper()}")
    print(f"Average Model Inference Speed: {avg_fps:.2f} Frames/Second")
    if valid_masks > 0:
        print(f"Average Inference Precision (mIoU): {total_miou / valid_masks:.4f} across {valid_masks} matched ground-truths")
    print("================================================")
    print(f"All inference diagnostic images safely exported to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drivable area inference over new generic data")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing new input images/dashcam frames")
    parser.add_argument("--output_dir", type=str, default="c:/drivableseg/inference_new_data_results", help="Directory to save output overlays")
    parser.add_argument("--model", type=str, default="c:/drivableseg/drivable_model_latest.pth", help="Path to model weights checkpoint")
    parser.add_argument("--limit", type=int, default=30, help="Maximum number of images to process")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional ground-truth mask directory to enable real-time mIoU tracking analysis")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix filter for isolating generic splits (e.g. 'val_')")
    
    args = parser.parse_args()
    run_inference(args.input_dir, args.output_dir, args.model, args.limit, args.mask_dir, args.prefix)
