# 🚗 Drivable Space Segmentation Pipeline - Project Documentation

This document serves as a comprehensive log of all the architectural shifts, deep learning pipeline integrations, and optimization algorithms constructed during the system overhaul to improve the real-time inference efficiency and push the mathematical `mIoU` bounds toward the **85% Target**.

---

## 1. Tri-State Classification Expansion
**Objective:** Transition the baseline binary model (Road vs Non-Road) into a strict priority-based 3-Class architecture. 
- **Dataset Re-Routing:** Modified the `export_idd_dataset.py` logic to emit exact 3-class target integer maps (`Background=0`, `High Priority Road=1`, `Low Priority Drivable Fallback=2`).
- **Mathematical Multi-Layering:** Upgraded `offline_dataset` to load raw `torch.long` Tensors directly instead of utilizing binary float normalization mapping sequences.
- **Loss Function Overhaul:** Built a `MultiClassDiceFocalLoss` custom object that merges standard Categorical Cross-Entropy with mathematically robust One-Hot Encoded Dice algorithms to battle class-imbalances.

## 2. Real-Time Benchmarking & Visualizations Pointers
**Objective:** Correctly wire the physical diagnostics, metrics tracing, and output matrices safely.
- **Inference Extrapolation:** Updated `infer_pipeline.py` to decode raw tri-state logits using `torch.argmax` logic, explicitly isolating the priority classes into native `Green/Blue` overlay visualizations natively onto dashcam imagery!
- **True GPU Synchronous Profiling:** Benchmarked PyTorch async routines incorrectly reporting physical execution times visually overhead (`Avg FPS: ~50` vs visually `120+`). Corrected this by applying active `torch.cuda.synchronize()` blocks alongside a native CUDA initialization warmup loop!
- **Hardware Integration Analysis:** Attached real-time automatic `mIoU` logic straight to the visualizer, allowing manual inference runs targeting existing target directories to immediately report the exact categorical metric overlaps (average precision!).
- **Graphs Integration:** Built clean Pyplot metric artifacts automatically saving directly into a distinct `/training_graphs/` directory.

## 3. Parallel "Heavy Architecture" Branching (EfficientNet)
**Objective:** Utilize massive surplus FPS to forcefully increase the network parameters and chase the extreme 85% mIoU targets.
- **`model_effnet.py` Construction:** Completely engineered a brand-new, mathematically rigid **EfficientNet-B2 variant** from the ground up since pre-trained frameworks were strictly banned.
  - Implemented custom `SqueezeExcitation` blocks to organically turn down "noisy" map channels dynamically.
  - Replaced standard jagged `ReLU6` outputs with smooth stochastic `SiLU` activations to prevent mathematical flattening during backpropagation.
  - Mapped a perfect output stride configuration explicitly connecting deeply to a generic custom `DeepLabV3+` Spatial Pyramid Pooling head.
- **`train_effnet.py` Target Script:** Created a fully independent training framework specifically locking the math away into `drivable_model_effnet.pth` to entirely evade corrupting the older, stable 78% `MobileNetV2` baseline checkpoint mathematically.

## 4. Aggressive Spatial Augmentation Combat Engine
**Objective:** The 8-Million-Parameter Heavy Model successfully broke the MobileNet ceiling out of the gate (~76.4% mIoU in 14 Epochs) but inevitably fell into "Memorization Traps" due to lack of diverse target material parameters.
- **`offline_dataset_aug.py` Module:** Engineered an isolated chaotic generator. Natively implements explicit `torchvision.transforms.functional.affine` geometric stretching logic targeting input batches automatically.
  - Implements massive physical shifts (-10 to 10 degree rotations) and zooming factors perfectly scaled. 
  - Overrides geometric constraints mapping the categorical target layer directly onto `NEAREST`-neighbor constraints, guaranteeing the `0,1,2` logic does not mathematically average to corrupt floats (e.g., `1.6`) when twisted sideways. 
- **`train_effnet_aug.py` Jumpstart Control:** Fully wired to seamlessly locate prior static `train_effnet` artifacts, inject them directly as a jumping-off point, and mathematically loop the AI against extreme graphical distortion parameters without overwriting legacy file branches.
