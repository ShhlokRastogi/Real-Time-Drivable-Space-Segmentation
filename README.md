# 🚗 Real-Time Drivable Space Segmentation

<p align="center">
  <img src="https://img.shields.io/github/stars/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/github/forks/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Speed-77FPS-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/mIoU-90.17%25-blue?style=for-the-badge" />
</p>

---

## 🔥 TL;DR (For Recruiters)
**Real-time semantic segmentation system achieving 90.17% mIoU at ~78 FPS using EfficientNet-B2 + DeepLabV3+ on a merged multi-domain driving dataset (NuScenes + IDD).**

- 🚀 Production-oriented (real-time constraint)
- 🧠 Strong generalization across road types
- ⚡ Optimized for consumer GPUs
- 📦 Plug-and-play inference

---

## 📌 Project Overview
This project builds a **real-time drivable space segmentation system** capable of handling:
- Unstructured rural roads
- Urban traffic environments
- Highways with lane markings

Unlike standard models, this system prioritizes **latency + accuracy together**, making it suitable for **autonomous driving pipelines and edge deployment**.

---

## 🧠 Model Architecture

| Component | Design Choice |
|----------|-------------|
| Encoder | EfficientNet-B2 |
| Decoder | DeepLabV3+ (ASPP) |
| Loss | Dice + Focal Loss |
| Framework | PyTorch |

### Why this works:
- EfficientNet → lightweight + strong features  
- DeepLabV3+ → multi-scale spatial understanding  
- Dice + Focal → handles class imbalance + improves boundaries  

---

## 📊 Dataset Strategy

### Merged Road Dataset
- NuScenes
- India Driving Dataset (IDD)

### Key Engineering Decisions:
- Strict Train / Val / Test split (no leakage)
- Heavy geometric augmentations
- Cross-domain learning (India + global roads)

➡️ Result: Model learns **road semantics, not dataset bias**

---

## ⚙️ Setup

```bash
git clone https://github.com/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation.git
cd Real-Time-Drivable-Space-Segmentation
pip install -r requirements.txt
```

---

## ▶️ Run Inference

### 🎥 Video
```bash
python inference_video.py --video "input.mp4"
```

### 🖼️ Images
```bash
python inference_images.py --folder "./images"
```

---

## 📈 Results

| Metric | Value |
|------|------|
| mIoU | **90.17%** |
| FPS | **~77.98 FPS** |
| Test Loss | `0.0559` |
| Test Samples | 1,052 |

<img width="1113" height="488" alt="image" src="https://github.com/user-attachments/assets/82534aa0-07b0-4304-aeca-113b1a0d4645" />

---

## 🎯 Why This Project Stands Out

- ⚡ **Real-time + high accuracy (rare combination)**
- 🌍 **Generalizes across countries (IDD + NuScenes)**
- 🧪 **Proper ML pipeline (no leakage, strong validation)**
- 🛠️ **Custom loss engineering**
- 📦 **Ready-to-use inference system**

---

## 📁 Outputs
```
/inference_effnet_test_results/
```

---

## 📦 Model Weights
```
drivable_model_effnet_merged_epoch11.pth
```

---

## 🤝 Contributions
Contributions, issues, and feature requests are welcome!

---

## ⭐ If you like this project
Give it a ⭐ on GitHub — it helps visibility!
