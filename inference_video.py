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
    print(f"\n--- Firing up Real-Time Dashcam Inference on {device} ---")
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
    cv2.putText(overlay, f"SysFPS: {fps_val:.1f} | Dashcam Sync", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    return overlay

def infer_video(vid_path, model, device, out_dir):
    print(f"Binding Video Stream Target: {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error failing to burst video structure: {vid_path}")
        return
        
    base_name = os.path.basename(vid_path)
    out_path = os.path.join(out_dir, f"AI_{base_name}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        overlay = process_frame(frame, model, device)
        writer.write(overlay)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed Frame Sync {frame_idx}...")
            
    cap.release()
    writer.release()
    print(f"\nTotal Video Logic Secured -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Public Dashcam Video CPU/GPU Inference.")
    parser.add_argument("--model", type=str, default="drivable_model_effnet_merged.pth", help="Path to your downloaded .pth weights.")
    parser.add_argument("--video", type=str, required=True, help="Path to your raw MP4/AVI dashcam stream.")
    parser.add_argument("--output", type=str, default="./video_output", help="Directory for the rendered MP4 output.")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    device = configure_device()
    model = load_model(args.model, device)
    
    infer_video(args.video, model, device, args.output)

if __name__ == "__main__":
    main()
