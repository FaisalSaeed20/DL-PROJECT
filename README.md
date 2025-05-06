# 🔥 Forest Fire and Smoke Detection Using Deep Learning

This repository contains the complete implementation of a deep learning-based system for detecting forest fires and smoke using object detection, super-resolution enhancement, classification, and zero-shot learning. The goal is to build a scalable, intelligent, and deployable early warning system that can be used in real-time forest monitoring scenarios.

---

## 📌 Project Overview

The project uses a multi-stage pipeline for detecting and classifying fire and smoke in natural environments:

1. YOLOv8 is used for high-speed object detection to locate fire/smoke regions in forest surveillance images.
2. ESRGAN (Enhanced Super-Resolution GAN) improves the quality of detected region patches for better classification.
3. MobileNetV2 classifies cropped and enhanced regions into either fire or smoke.
4. CLIP (Contrastive Language-Image Pretraining) is used for zero-shot classification and image-to-text matching.
5. Streamlit frontend makes the system user-accessible for real-time detection.

---

## 🎯 Key Features

| Component       | Functionality                                                                 |
|----------------|--------------------------------------------------------------------------------|
| YOLOv8          | Object detection for fire/smoke regions                                        |
| ESRGAN          | Super-resolves cropped image patches for higher clarity                       |
| MobileNetV2     | Lightweight binary classification (fire vs. smoke)                             |
| CLIP            | Zero-shot prompt-based classification + image-to-text/image search             |
| Streamlit App   | Upload an image and visualize detection/classification results instantly       |

---

## 🧪 Improvements & Experiments

| Area                 | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| Dataset Expansion    | Included more diverse viewpoints (distant tower and close-up), different weather/light conditions |
| Synthetic Data       | Introduced Minecraft-generated fire scenes to improve generalization                             |
| ESRGAN Integration   | Enhanced YOLO-detected crops before classification for improved accuracy                         |
| Zoomed Cropping      | Zoomed and resized image patches before feeding into classifier                                  |
| CLIP Text Search     | Added capability to classify and search fire/smoke images using natural language prompts         |

---

## 📊 Results

| Task                     | Model          | Accuracy | Speed          | Notes                             |
|--------------------------|----------------|----------|----------------|-----------------------------------|
| Object Detection         | YOLOv8s        | 85%      | Real-time      | Fast enough for live surveillance |
| Classification (Fire/Smoke) | MobileNetV2 | 89.5%    | ~12ms per image| Suitable for edge deployment      |
| Zero-Shot Matching       | CLIP (ViT-B/32)| N/A      | Fast           | Used for prompt-based inference   |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/FaisalSaeed20/DL-PROJECT.git
cd DL-PROJECT
