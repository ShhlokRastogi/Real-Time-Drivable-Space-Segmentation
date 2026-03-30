# 🚗 Real-Time Drivable Space Segmentation

<p align="center">
  <img src="https://img.shields.io/github/stars/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/github/forks/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Speed-~78FPS-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/mIoU-90.17%25-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Semantic%20Segmentation-orange?style=for-the-badge" />
</p>

---

## 📌 Project Overview

This project implements a **real-time drivable space segmentation system** designed for **autonomous driving and edge deployment**.

The model accurately detects drivable regions across:
- Urban roads  
- Rural/unstructured roads  
- Highways  

It is optimized for **both speed and accuracy**, achieving:
- ⚡ ~78 FPS  
- 🎯 90.17% mIoU  

---

## 🧠 Model Architecture

- **Encoder:** EfficientNet-B2  
- **Decoder:** DeepLabV3+ (ASPP)  
- **Loss Function:** Dice Loss + Focal Loss  
- **Framework:** PyTorch  

### 🔍 Why this architecture?
- EfficientNet → lightweight and efficient  
- DeepLabV3+ → strong multi-scale segmentation  
- Dice + Focal → handles class imbalance and improves boundaries  

---

## 📊 Dataset Used

- **NuScenes Dataset**  
- **India Driving Dataset (IDD)**  

### Key Strategies:
- Proper train/validation/test split (no data leakage)  
- Heavy data augmentation  
- Cross-domain training for better generalization  

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/ShhlokRastogi/Real-Time-Drivable-Space-Segmentation.git
cd Real-Time-Drivable-Space-Segmentation
pip install -r requirements.txt
