# 🚗 Real-Time Drivable Space Segmentation

## Problem Statement

The objective of this project is to build a mathematically rigorous and highly accurate (targeting `>85% mIoU`) computer vision pipeline capable of detecting general drivable boundaries strictly within a **real-time execution window** (`>60 FPS`) on local consumer hardware.

Real-world driving environments have enormous variance—ranging from completely unlined rural roads to brightly structured highways. To ensure the artificial network generalizes rather than memorizes, we actively compiled **BDD100k, NuScenes, and the India Driving Dataset (IDD)** into a massive, centralized `merged_road_dataset`.

To operate within our aggressive latency constraints while matching heavy parameter requirements, the network infrastructure utilizes a purely hand-engineered **EfficientNet-B2** scaling encoder paired forcefully with a **DeepLabV3+** Spatial Pyramid Pooling decoder algorithm to capture both microscopic edge boundaries and macroscopic structural context.

---

## Dataset & Training Strategy (Merged Road Context)

The compilation of the **Merged Road Dataset** relies heavily on aggressively isolating the data into strict Train, Validation, and unseen Test subsets purely to defeat data leakage. During the actual epoch loop (`train_effnet_merged.py`), the `MergedRoadDataset` torch loader intrinsically binds severe matrix augmentations (`torchvision.transforms.functional.affine`) across spatial planes to systematically stress the geometry learning cap.

The network optimization relies entirely on an isolated `MultiClassDiceFocalLoss` engine designed deliberately to mathematically crush precision failures on harsh categorical boundary imbalances.

### Network Progression Mapping

Below is the graph visualization mapping out the model's combinatorial loss trajectory and evaluation limits natively during the execution cycle:

![Training Metrics](/c:/drivableseg/training_graphs/effnet_merged_training_metrics_graph.png)

---

## Final Mathematical Evaluation (Test Split)

The stabilized system weights (`drivable_model_effnet_merged.pth`) were extracted and benchmarked algorithmically strictly against `1,052` entirely unseen, un-augmented graphical frames living randomly in the isolated `test/` boundaries of the merged road hub.

| Physical Metric | Recorded Result | System Target | Project Status |
| :--- | :--- | :--- | :--- |
| **Hardware Used** | PyTorch CUDA Tensor Cores | *N/A* | ✅ Active |
| **Total Test Split Loss** | `0.0559` | *Minimized* | 📉 Converged |
| **Generalized mIoU** | **`90.17%`** | `>= 85.00%` | 🏆 Surpassed Target Bounds |
| **Execution Speeds** | **`~77.98 FPS`** | `>= 60.0 FPS` | ⚡ Real-Time Inference Success |

Visual artifact matrices and diagnostic dashcam streams for these 1,000+ unseen boundaries run by the AI brain were exclusively dumped into the native `/inference_effnet_test_results/` system directory.
