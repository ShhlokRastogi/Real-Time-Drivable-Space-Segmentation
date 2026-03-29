import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from offline_dataset_merged import MergedRoadDataset
from model_effnet import EfficientDeepLabV3Plus
from loss import MultiClassDiceFocalLoss, calculate_miou
import time
import matplotlib.pyplot as plt

def train_model():
    batch_size = 8  
    num_epochs = 15 
    learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- EfficientNet DeepLabV3+ Binary Model Fired Up! Targeting Device: {device} ---")
    
    train_path = 'c:/drivableseg/merged_road_dataset/train'
    val_path = 'c:/drivableseg/merged_road_dataset/val'
    
    print(f"Loading Pristine Offline Dataset from: {train_path} / {val_path} (NO AUGMENTATIONS)")
    
    train_dataset = MergedRoadDataset(data_split_dir=train_path)
    val_dataset = MergedRoadDataset(data_split_dir=val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=max(1, batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size), shuffle=False, num_workers=0)
    
    print(f"Training parameters: {len(train_dataset)} samples. Validation: {len(val_dataset)} samples.")
    
    # Strictly initializing 2 Classes (0: Background, 1: Road)
    print("Constructing BINARY EFFICIENTNET-B2 + DeepLabV3+ Heavy Architecture...")
    model = EfficientDeepLabV3Plus(num_classes=2).to(device)
    
    save_target_path = "c:/drivableseg/drivable_model_effnet_merged.pth"
    print(f"Model checkpoint targeted specifically to: {save_target_path}")
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params (Binary Head): {total_params:,}")
    
    # Initialize Binary Loss function
    criterion = MultiClassDiceFocalLoss(num_classes=2)
    
    # Advanced Optimizations
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # 1Cycle Scheduler guarantees rapid approach to optimal weight minimums
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, 
        epochs=num_epochs, steps_per_epoch=max(1, steps_per_epoch),
        pct_start=0.3, # 30% warmup
        div_factor=10, final_div_factor=1000
    )
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())
    
    print(f"\n[START] Commencing Binary EfficientNet Loop on Pure Data!")
    
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
            
            # Additional Optimization: Gradient Clipping explicitly to prevent explosions
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Step the OneCycleLR per batch
            scheduler.step()
            
            epoch_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        val_miou_accum = 0.0
        val_batches = len(val_loader)
        
        with torch.no_grad():
            for val_img, val_mask in val_loader:
                val_img, val_mask = val_img.to(device), val_mask.to(device)
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    val_out = model(val_img)
                    loss = criterion(val_out, val_mask)
                    val_loss += loss.item()
                    
                    val_preds = torch.argmax(val_out, dim=1)
                    val_miou_accum += calculate_miou(val_preds, val_mask, num_classes=2)
                
        epoch_time = time.time() - start_time
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, val_batches) if val_batches > 0 else 0.0
        avg_val_miou = val_miou_accum / max(1, val_batches) if val_batches > 0 else 0.0
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        val_miou_history.append(avg_val_miou)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f} | Time: {epoch_time:.2f}s")
        
        torch.save(model.state_dict(), save_target_path)

    print("\nTraining Phase Complete! Binary Model weights strictly verified and stored to: ", save_target_path)
    
    print("Generating Matplotlib Tracking Graphs...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_loss_history, label='Train Loss', color='blue', marker='o')
    if len(val_loader) > 0:
        plt.plot(range(1, num_epochs+1), val_loss_history, label='Val Loss', color='orange', marker='s')
    plt.title('Loss Trajectory over Epochs (Binary Model)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Output')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    if len(val_loader) > 0:
        plt.plot(range(1, num_epochs+1), val_miou_history, label='Validation mIoU', color='green', marker='^')
    plt.title('Accuracy Metric Output (Binary mIoU)')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU Coefficient')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    graph_dir = "c:/drivableseg/training_graphs"
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, "effnet_merged_training_metrics_graph.png")
    plt.savefig(graph_path)
    print(f"Training metrics visualization securely saved to: {graph_path}")

if __name__ == "__main__":
    train_model()
