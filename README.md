# DL-PROJECT
# 🔥 Forest Fire and Smoke Detection using Deep Learning

This project presents a complete deep learning pipeline for detecting forest fires and smoke using YOLOv8 for object detection, ESRGAN for image super-resolution, MobileNetV2 for region classification, and CLIP for zero-shot image-text understanding. A user-friendly Streamlit web application has also been developed for real-time fire and smoke detection.

---


---

## 🚀 Key Components

### 🔍 YOLOv8 Detection
- Detects fire and smoke regions in images using bounding boxes.
- Trained on a custom dataset with forest fire and smoke images.
- Achieves 85% detection accuracy.

### 🧠 ESRGAN Super-Resolution
- Converts YOLO-cropped image patches into high-resolution versions using Real-ESRGAN (RRDBNet).
- Helps preserve fine-grained features like smoke trails and flames for better classification.

### 🔬 MobileNetV2 Classification
- Lightweight classifier trained to distinguish between fire and smoke in cropped patches.
- Accuracy: 89.5%
- Inference Speed: ~12ms per image
- Optimized for edge deployment.

### ✨ CLIP Zero-Shot Learning
- Uses OpenAI's CLIP (ViT-B/32) to classify and retrieve fire/smoke images based on text prompts.
- Supports zero-shot classification and image-text search.

### 🌐 Streamlit Web App
- Upload an image and detect fire/smoke in real time.
- Shows bounding boxes, confidence scores, and classification results.
- Easy to use and lightweight.

---

## 🧪 Performance Summary

| Model         | Task                       | Accuracy | Inference Speed |
|---------------|----------------------------|----------|-----------------|
| YOLOv8s       | Fire/Smoke Detection       | 85%      | Real-time       |
| MobileNetV2   | Cropped Region Classification | 89.5%   | ~12ms/image     |
| CLIP          | Zero-Shot Prompt Matching  | N/A      | Fast            |

---

## 🛠 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/FaisalSaeed20/DL-PROJECT.git
cd DL-PROJECT
---
## Running the StreamLit App
cd streamlit_app
streamlit run app.py


