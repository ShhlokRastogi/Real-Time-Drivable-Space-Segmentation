import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from offline_dataset import OfflineNuScenesDataset
from model_effnet import EfficientDeepLabV3Plus
from loss import MultiClassDiceFocalLoss, calculate_miou
import time
import matplotlib.pyplot as plt

def train_model():
    # Because EfficientNet is "heavier", we might have to be slightly careful with memory,
    # but 8 is usually fine.
    batch_size = 8  
    num_epochs = 15 # Trimmed back down to 15 for faster initial prototyping
    learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Heavy Deep Learning Engine Fired Up! Targeting Device: {device} ---")
    
    data_path = 'c:/drivableseg/exported_data_idd20k'
    print(f"Loading 640x360 Pre-Rendered Offline Dataset from: {data_path}")
    
    try:
        full_dataset = OfflineNuScenesDataset(data_dir=data_path, augment=True)
    except Exception as e:
        print("Warning: Dataset not safely found.")
        full_dataset = [ (torch.zeros((3, 360, 640)), torch.zeros((360, 640), dtype=torch.long)) for _ in range(16) ]

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if train_size == 0 and len(full_dataset) > 0: train_size, val_size = len(full_dataset), 0
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=max(1, batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size), shuffle=False, num_workers=0) if val_size > 0 else []
    
    print(f"Training split: {train_size} samples. Validation split: {val_size} samples.")
    
    print("Constructing EFFICIENTNET-B2 + DeepLabV3+ Heavy Architecture (No Pre-Trained Weights)...")
    model = EfficientDeepLabV3Plus(num_classes=3).to(device)
    
    # Target our dedicated checkpoint so we do not overwrite the MobileNetV2 version
    checkpoint_path = "c:/drivableseg/drivable_model_effnet_latest.pth"
    if os.path.exists(checkpoint_path):
        print(f"Resuming EfficientNet training from existing weights: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Base Parameters: {total_params:,}")
    
    criterion = MultiClassDiceFocalLoss(num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())
    
    print(f"\n[START] Commencing EfficientNet Epoch Analysis Loop!")
    
    train_loss_history = []
    val_loss_history = []
    val_miou_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        val_miou_accum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_img, val_mask in val_loader:
                val_img, val_mask = val_img.to(device), val_mask.to(device)
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    val_out = model(val_img)
                    loss = criterion(val_out, val_mask)
                    val_loss += loss.item()
                    
                    val_preds = torch.argmax(val_out, dim=1)
                    val_miou_accum += calculate_miou(val_preds, val_mask, num_classes=3)
                    val_batches += 1
                
        epoch_time = time.time() - start_time
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, val_batches) if val_batches > 0 else 0.0
        avg_val_miou = val_miou_accum / max(1, val_batches) if val_batches > 0 else 0.0
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        val_miou_history.append(avg_val_miou)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f} | Time: {epoch_time:.2f}s")
        
        torch.save(model.state_dict(), checkpoint_path)
        scheduler.step()

    print("\nTraining Phase Complete! Production Model weights securely stored to: ", checkpoint_path)
    
    print("Generating EfficientNet Matplotlib Tracking Graphs...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_loss_history, label='Train Loss', color='blue', marker='o')
    if len(val_loader) > 0:
        plt.plot(range(1, num_epochs+1), val_loss_history, label='Val Loss', color='orange', marker='s')
    plt.title('Loss Trajectory over Epochs (EfficientNet)')
    plt.xlabel('Epochs')
    plt.ylabel('Combinatorial Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(val_loader) > 0:
        plt.plot(range(1, num_epochs+1), val_miou_history, label='Validation mIoU', color='green', marker='^')
    plt.title('Accuracy Metric Output (mIoU)')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU Coefficient')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    graph_dir = "c:/drivableseg/training_graphs"
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, "effnet_training_metrics_graph.png")
    plt.savefig(graph_path)
    print(f"Training metrics visualization securely saved to: {graph_path}")

if __name__ == "__main__":
    train_model()
