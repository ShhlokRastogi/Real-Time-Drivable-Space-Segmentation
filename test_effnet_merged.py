import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from model_effnet import EfficientDeepLabV3Plus
from loss import MultiClassDiceFocalLoss, calculate_miou

import argparse

def run_testing(model_path, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Firing up Real-Time Metric Inference Pipeline on {device} ---")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find checkpoint at {model_path}.")
        return
        
    print(f"Loading Binary EfficientNet-B2 Topology into VRAM securely from: {model_path}")
    model = EfficientDeepLabV3Plus(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    # [WARMUP PHASE]
    if device.type == 'cuda':
        print("Warming up CUDA cores for pristine benchmark stabilization...")
        dummy = torch.randn(1, 3, 360, 640).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize()
    
    input_dir = 'c:/drivableseg/merged_road_dataset/test/images'
    mask_dir = 'c:/drivableseg/merged_road_dataset/test/masks'
    os.makedirs(out_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    total_time = 0.0
    total_miou = 0.0
    total_loss = 0.0
    valid_masks = 0
    
    criterion = MultiClassDiceFocalLoss(num_classes=2)
    
    print(f"Running automated statistical inference on {len(img_files)} totally unseen frames...")
    
    with torch.no_grad():
        for i, img_name in enumerate(img_files):
            img_path = os.path.join(input_dir, img_name)
            
            # Use exact mathematical logic from offline_dataset_merged.py
            pil_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = pil_img.size
            img_resized = pil_img.resize((640, 360), resample=Image.BILINEAR)
            
            img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # FPS BENCHMARK START
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    raw_pred = model(img_tensor)
                torch.cuda.synchronize() 
            else:
                raw_pred = model(img_tensor) 
            
            # Binary Argmax Collapse
            class_pred_tensor = torch.argmax(raw_pred, dim=1)
            class_pred = class_pred_tensor.squeeze().cpu().numpy().astype(np.uint8)
            
            end_time = time.time()
            # FPS BENCHMARK END
            
            frame_time = end_time - start_time
            total_time += frame_time
            
            # Scale prediction back mathematically
            class_pred_resized = cv2.resize(class_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # GROUND TRUTH MATH
            base_name = img_name.rsplit('.', 1)[0]
            mask_path_png = os.path.join(mask_dir, base_name + '.png')
            mask_path_jpg = os.path.join(mask_dir, base_name + '.jpg')
            m_path = mask_path_png if os.path.exists(mask_path_png) else mask_path_jpg
            
            current_miou_str = ""
            if os.path.exists(m_path):
                gt_mask_pil = Image.open(m_path).convert('L')
                gt_mask_resized = gt_mask_pil.resize((640, 360), resample=Image.NEAREST)
                
                gt_tensor = torch.from_numpy(np.array(gt_mask_resized)).long()
                gt_tensor = torch.clamp(gt_tensor, 0, 1).unsqueeze(0).to(device)
                
                # Active Loss calculation
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        frame_loss = criterion(raw_pred, gt_tensor).item()
                else:
                    frame_loss = criterion(raw_pred, gt_tensor).item()
                    
                total_loss += frame_loss
                
                # Binary mIoU calculation natively matching the train loop
                calc_val = calculate_miou(class_pred_tensor, gt_tensor, num_classes=2)
                total_miou += calc_val
                valid_masks += 1
                current_miou_str = f" | mIoU: {calc_val:.2f}"
            
            # VIZ GENERATOR
            orig_img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            color_mask = np.zeros_like(orig_img_cv)
            
            # Fill Road with vibrant Green [B, G, R]
            color_mask[class_pred_resized == 1] = [0, 255, 0] 
            overlay = cv2.addWeighted(orig_img_cv, 1.0, color_mask, 0.4, 0)
            
            fps_val = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(overlay, f"FPS: {fps_val:.1f}{current_miou_str}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            out_path = os.path.join(out_dir, f"AI_pred_{img_name}")
            cv2.imwrite(out_path, overlay)
            
            if (i+1) % 50 == 0:
                print(f"Processed {i+1}/{len(img_files)} Test Set frames...")

    avg_fps = len(img_files) / total_time if total_time > 0 else 0
    final_miou = total_miou / valid_masks if valid_masks > 0 else 0
    final_loss = total_loss / valid_masks if valid_masks > 0 else 0
    
    print("\n================================================")
    print("[FINAL METRICS] TEST SPLIT EVALUATION [RESULTS]")
    print("================================================")
    print(f"Target Hardware Configuration: {device.type.upper()}")
    print(f"Total Test Set Combinatorial Loss: {final_loss:.4f}")
    if valid_masks > 0:
        print(f"Final Generalized Inference mIoU:  {final_miou:.4f} ({final_miou*100:.2f}%)")
    print(f"Average Real-Time Execution Speed: ~{avg_fps:.2f} FPS")
    print("================================================")
    print(f"All graphic inference arrays securely rendered to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time EffNet Metrics")
    parser.add_argument("--model", type=str, default="c:/drivableseg/drivable_model_effnet_merged.pth", help="Path to weights")
    parser.add_argument("--out", type=str, default="c:/drivableseg/inference_effnet_test_results", help="Directory for output renders")
    
    args = parser.parse_args()
    run_testing(args.model, args.out)
