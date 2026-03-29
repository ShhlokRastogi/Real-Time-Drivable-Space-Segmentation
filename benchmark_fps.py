import torch
import time
from model import RealTimeDeepLabV3Plus

def benchmark_fps(model, device='cuda', input_size=(1, 3, 512, 1024), num_iterations=100):
    model.to(device)
    model.eval()
    
    # Dummy input
    x = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
            
    if device == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)
            
    if device == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    print(f"FPS: {fps:.2f}, Avg Time per Frame: {avg_time*1000:.2f} ms")

if __name__ == '__main__':
    model = RealTimeDeepLabV3Plus(num_classes=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on {device}")
    benchmark_fps(model, device=device)
    
    # Print model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")
