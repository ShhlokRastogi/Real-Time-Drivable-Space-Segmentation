# AI HANDOVER CONTEXT

**Project Goal**: Real-Time Multi-Class Drivable Space Segmentation
**Primary Targets**: `>= 85% mIoU` while strictly maintaining `>= 60 FPS` inference speed.
**Hardware Profile**: CUDA GPU available.

## Hard Constraints
1. **NO PRE-TRAINED WEIGHTS ALLOWED.** All models must be built and trained strictly from scratch.
2. Cannot use massive networks (e.g., standard ResNet-101/Vision Transformers) due to the strict 60 FPS inference latency constraint. Approved backbone families are MobileNet and EfficientNet.

## Categorical Logic (Tri-State System)
The project transitioned from binary to a 3-class priority logic system:
- **Class 0**: Background
- **Class 1**: High-Priority Road (Mapped to Green in Inference)
- **Class 2**: Low-Priority Drivable Fallback (Mapped to Blue in Inference)
*Note: Matrices rely on raw integer classes mapping `[0, 1, 2]`. No normalized float tensors are passed into the categorical cross-entropy backend.*

---

## Architectural Branches

The repository has two perfectly isolated parallel pipelines. DO NOT overwrite the V1 files with the V2 concepts. They exist simultaneously to protect the baseline limits.

### V1: Baseline MobileNet Pipeline (Safe Fallback)
- **Current Status**: Maxed out at `~78.06% mIoU`. Began overfitting due to low spatial parameter caps (1.9M params). Highly performant (`>120 FPS`).
- **Files**:
  - `model.py`: Lightweight `MobileNetV2` + `DeepLabV3+` Decoder. Uses InvertedResidual blocks with ReLU6.
  - `offline_dataset.py`: Standard IDD data ingestion. Applies only basic horizontal flips and color jittering.
  - `train.py`: Standard PyTorch harnessing `CosineAnnealingLR`. Outputs to `drivable_model_latest.pth`.

### V2: Heavy Augmented EfficientNet Pipeline (Current Active Target)
- **Current Status**: Built to specifically crush the 85% mIoU goal. Training natively in the background. Vastly heavier capacity (`~8M params`).
- **Files**: 
  - `model_effnet.py`: Hand-built `EfficientNetEncoder` mapping `MBConv` blocks. Utilizes `SqueezeExcitation` Algorithmic Attention maps and `SiLU` (Swish) activations to smoothly calculate extreme depths safely. Output hooked to a 1280-Channel `DeepLab` ASPP.
  - `offline_dataset_aug.py`: Chaotic Geometry Engine. Natively executes `torchvision.transforms.functional.affine`. Randomly applies severe Rotations (-10 to 10 deg), Translations, and Scale limits. **CRITICAL**: The image relies on `BILINEAR` interpolation, but the mask strictly enforces `NEAREST` neighbor mapping to prevent `[0, 1, 2]` integers from corrupting into float coordinates during tilting.
  - `train_effnet_aug.py`: Jumpstart trainer natively loads from V1's `pth` artifacts to gain momentum, but strictly saves its advanced logic isolated into `drivable_model_effnet_aug.pth`. 

---

## Key Algorithmic Mechanics

1. **Loss Infrastructure**: Handled by `MultiClassDiceFocalLoss` inside `loss.py`. It combines `CrossEntropy` with a One-Hot encoded `Dice Loss` calculator explicitly tuned to handle the severe pixel-area imbalance of the Background vs Drivable boundaries.
2. **Inference Benchmarking**: `infer_pipeline.py` features a custom Argmax bounding extraction engine.
   - For real-world benchmark readings, it utilizes explicit `torch.cuda.synchronize()` blocks surrounding the `model(input)` forward pass to defeat Python's asynchronous kernel queuing illusions (which previously misreported 60 FPS frames as 800 FPS).
   - Can optionally calculate mIoU dynamically via the `--mask_dir` terminal flag to cross-reference AI accuracy natively without needing a full training suite.

## AI Agent Directives
If picking up this workflow, rely entirely on scaling the `train_effnet_aug.py` branch further to hit the 85% mark. Do not downgrade back to `MobileNet`. Focus on hyperparameter tuning learning rates or tweaking the `DiceFocalLoss` alphas/gammas if convergence stalls past Epoch 40.
