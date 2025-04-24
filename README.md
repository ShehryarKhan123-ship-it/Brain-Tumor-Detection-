# 🧠 Brain Tumor Detection & Segmentation using YOLOv11 and SAM

## Demo Video
https://drive.google.com/file/d/1aDn-vZOcogTQCCG6JppfSPqDvLDa5j20/view?usp=sharing


## 📌 Overview

This repository showcases a deep learning pipeline for **brain tumor detection and segmentation** using two state-of-the-art models:  
- **YOLOv11** for real-time object detection  
- **Segment Anything Model (SAM)** for pixel-level image segmentation  

The fusion of these powerful tools enables the **accurate identification** and **precise segmentation** of brain tumors from medical imaging data such as MRI scans. Early and reliable detection of tumors is vital for successful treatment, and this project contributes toward automating that process.

---

## ✨ Key Features

- 🔍 **High-Precision Detection**: YOLOv11 identifies tumor locations with optimized speed and accuracy.
- 🎯 **Pixel-Accurate Segmentation**: SAM refines YOLO-detected regions into detailed segmentation masks.
- 🔄 **End-to-End Workflow**: From raw data to model inference, the complete pipeline is included.
- 📁 **Custom Dataset Ready**: Easily adaptable to any medical image dataset.
- 📚 **Readable Codebase**: Clean, modular, and well-documented Python code.

---

## ⚙️ Installation Guide

Follow these steps to get started:

### 1. Clone the Repository
```bash
git clone <repository_url>
cd brain-tumor-detection-yolov11-sam
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> 💡 Ensure CUDA and cuDNN are correctly installed if you intend to utilize GPU acceleration.

---

## 📊 Dataset Preparation

This project is compatible with standard medical imaging datasets. You are expected to format your dataset as follows:

### ✅ Data Requirements
- **Images**: MRI scans or relevant brain imaging files.
- **YOLOv11 Annotations**: Text files in YOLO format (`class_id center_x center_y width height`).
- **SAM Data**: Segmentation masks (optional if using SAM purely for inference).

### 📂 Directory Structure
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
```

### ⚙️ Configuration
Update `data.yaml` with:
```yaml
train: path/to/train/images
val: path/to/valid/images
nc: 1  # number of classes
names: ['tumor']
```

---

## 🧠 Model Architecture

### 🔸 YOLOv11
A high-performance object detection model that processes the full image in a single forward pass. It outputs bounding boxes around tumors.

### 🔹 Segment Anything Model (SAM)
Developed by Meta AI, SAM segments any object based on a given prompt. In this pipeline, SAM receives bounding box prompts from YOLOv11 outputs to isolate tumor regions at the pixel level.

### 🔄 Pipeline Flow:
1. **YOLOv11** detects tumor bounding boxes.
2. **SAM** receives bounding boxes as prompts.
3. **SAM** outputs segmentation masks for each detected tumor region.

---

## 🏋️ Model Training

### 1. Train YOLOv11
Customize `yolov11.yaml` as needed, then run:
```bash
python train_yolo.py --cfg yolov11.yaml --data data.yaml --epochs 100 --batch-size 16
```

### 2. SAM Integration
If using pre-trained SAM:
- No training required — SAM uses YOLO prompts at inference.

If fine-tuning SAM:
- Modify the segmentation scripts accordingly.
- Follow dataset-specific training procedures.

---

## 📈 Evaluation

### 🔍 YOLOv11 Metrics:
- **mAP**, **Precision**, **Recall**, **F1-score**
```bash
python val_yolo.py --weights runs/train/exp/weights/best.pt --data data.yaml
```

### 🧪 Segmentation Metrics:
- **IoU**, **Dice Coefficient**, **Pixel Accuracy**

For evaluating combined results from YOLO + SAM, refer to `evaluate_segmentation.py`.

---

## 🚀 Inference

Perform detection and segmentation on new medical images:
```bash
python detect_and_segment.py --source /path/to/image.jpg --yolo-weights runs/train/exp/weights/best.pt
```

The script:
- Runs YOLOv11 for detection
- Feeds detections to SAM
- Outputs segmented tumor masks

---

## 📦 Folder Structure (Simplified)

```
brain-tumor-detection-yolov11-sam/
├── data/
│   └── data.yaml
├── scripts/
│   ├── train_yolo.py
│   ├── detect_and_segment.py
│   └── evaluate_segmentation.py
├── runs/
│   └── train/exp/
├── requirements.txt
└── README.md
```

---

## 🧠 Credits

- **YOLOv11**: Developed by the YOLO research community.
- **SAM**: Created by [Meta AI Research](https://segment-anything.com).
- **MRI Datasets**: Acknowledge the dataset providers as per their license.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
