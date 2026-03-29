import os
import cv2
import time
import torch
import numpy as np
import argparse
from PIL import Image
from model_effnet import EfficientDeepLabV3Plus

def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Firing up Batch Folder Inference Engine on {device} ---")
    return device

def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Could not find model at {model_path}.")
    print(f"Loading EfficientNet-B2 Model natively from: {model_path}")
    model = EfficientDeepLabV3Plus(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    if device.type == 'cuda':
        print("Warming up CUDA tensor geometry...")
        dummy = torch.randn(1, 3, 360, 640).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for _ in range(5):
                model(dummy)
        torch.cuda.synchronize()
        
    return model

def process_frame(frame, model, device):
    orig_h, orig_w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_resized = pil_img.resize((640, 360), resample=Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    start_time = time.time()
    if device.type == 'cuda': torch.cuda.synchronize()
    
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                raw_pred = model(img_tensor)
        else:
            raw_pred = model(img_tensor)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    frame_time = time.time() - start_time
    
    class_pred_tensor = torch.argmax(raw_pred, dim=1)
    class_pred = class_pred_tensor.squeeze().cpu().numpy().astype(np.uint8)
    class_pred_resized = cv2.resize(class_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    color_mask = np.zeros_like(frame)
    color_mask[class_pred_resized == 1] = [0, 255, 0] 
    overlay = cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0)
    
    fps_val = 1.0 / frame_time if frame_time > 0 else 0
    cv2.putText(overlay, f"SysFPS: {fps_val:.1f} | Drivable Frame", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    return overlay

def infer_folder(folder_path, model, device, out_dir):
    print(f"\n[TARGET: FOLDER] Sweeping entirely through directory -> {folder_path}")
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
    
    if len(files) == 0:
        print(f"Error: No valid images found recursively in {folder_path}!")
        return
        
    for i, f in enumerate(files):
        img_path = os.path.join(folder_path, f)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Error decoding visual frame: {img_path}")
            continue
            
        overlay = process_frame(frame, model, device)
        
        base_name = os.path.basename(img_path)
        out_path = os.path.join(out_dir, f"AI_{base_name}")
        cv2.imwrite(out_path, overlay)
        if (i+1) % 10 == 0:
            print(f"Rendered {i+1} Frames -> {out_dir}")
            
    print(f"\nMatrix sweep fully secure. Extracted {len(files)} bounds to: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Public Image Folder Batch Inference Engine.")
    parser.add_argument("--model", type=str, default="drivable_model_effnet_merged_epoch11.pth", help="Path to your downloaded .pth weights.")
    parser.add_argument("--folder", type=str, required=True, help="Path pointing to a standard folder of raw images.")
    parser.add_argument("--output", type=str, default="./folder_output", help="Directory where the overlays are compiled.")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = configure_device()
    model = load_model(args.model, device)
    
    infer_folder(args.folder, model, device, args.output)

if __name__ == "__main__":
    main()
