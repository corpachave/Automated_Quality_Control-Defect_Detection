# VisionSpec QC - Automated PCB Defect Detection

A deep learning-based quality control system for automated PCB (Printed Circuit Board) defect detection. The system uses a two-stage approach: first classifying boards as PASS or DEFECT, then localizing specific defect types using object detection.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Models](#models)
- [API Documentation](#api-documentation)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements a computer vision system for automated PCB inspection in manufacturing environments. It combines:

- **Classification Model**: Binary classification (PASS/DEFECT) using transfer learning with MobileNetV2
- **Detection Model**: YOLOv8-based multi-class defect localization for 6 defect types

### Supported Defect Types

| Class | Description |
|-------|-------------|
| `open` | Open circuit (broken trace) |
| `short` | Short circuit (unintended connection) |
| `mouse_bite` | Missing copper portion |
| `spur` | Extra copper/solder |
| `pinhole` | Small hole in copper |
| `spurious` | Unwanted copper residue |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VisionSpec QC System                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐                     │ 
│  │   Streamlit  │────▶│   FastAPI    │                     │
│  │   Frontend   │     │   Backend    │                     │
│  └──────────────┘     └──────┬───────┘                     │
│                               │                             │
│                    ┌──────────┴──────────┐                  │
│                    ▼                     ▼                  │
│           ┌──────────────┐      ┌──────────────┐           │
│           │ Classification│      │  Detection   │           │
│           │    Model      │      │    Model     │           │
│           │ (MobileNetV2) │      │   (YOLOv8)   │           │
│           └──────────────┘      └──────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Components

- **Frontend**: Streamlit web UI for image upload, webcam capture, and results visualization
- **Backend**: FastAPI REST API for model inference
- **Classification Model**: TensorFlow/Keras MobileNetV2-based binary classifier
- **Detection Model**: Ultralytics YOLOv8 multi-class object detector

---

## Project Structure

```
Automated_Quality_Control-Defect_Detection/
├── app.py                    # Streamlit frontend application
├── main.py                   # FastAPI backend server
├── data.yaml                 # YOLO dataset configuration
│
├── train_model.py            # Classification model training script
├── train_detection.py        # Detection model training script
├── train_data_preparation.py # Data augmentation visualization
├── real_time_inference.py    # Live webcam/video inference
├── grad_cam.py               # Grad-CAM visualization for model interpretability
│
├── visionspec_qc_model.keras # Trained classification model
├── requirements.txt          # Python dependencies
├── start.ps1                 # PowerShell startup script
├── start.bat                 # Batch startup script
│
├── data/                     # Dataset directory
│   ├── classification/       # Binary classification data
│   │   ├── defect/           # Defect PCB images
│   │   └── pass/             # Good PCB images
│   └── detection/            # YOLO detection data
│       ├── images/
│       │   ├── train/        # Training images
│       │   └── val/          # Validation images
│       └── labels/           # YOLO format annotations
│
├── raw_data/                 # Original/raw datasets
│   ├── DeepPCB-master/       # DeepPCB dataset
│   ├── DsPCBSD/              # PCB defect dataset
│   └── PCB_Defect/           # Additional PCB dataset
│
├── runs/                     # Training outputs
│   └── detect/pcb_yolov8n/   # YOLO training results
│       └── weights/          # Trained detection weights
│
└── scripts/                  # Utility scripts
    ├── convert_data.py       # Data conversion utilities
    ├── convert_deepPCB.py    # DeepPCB format converter
    └── map_yolo_classes.py   # Class mapping utilities
```

---

## Features

### Core Functionality
-  Binary classification (PASS/DEFECT) with confidence score
-  Multi-class defect localization with bounding boxes
-  Real-time inference support (webcam/video)
-  Grad-CAM explainability visualization

### User Interface
-  Image upload for batch inspection
-  Webcam capture for live inspection
-  Visual detection results with bounding boxes
-  Detection confidence and labels display
-  Inference time tracking

### API
-  RESTful FastAPI endpoint
-  OpenAPI/Swagger documentation
-  JSON response with detections, status, and timing

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training, optional for inference)

### 1. Clone and Setup

```bash
# Navigate to project directory
cd Automated_Quality_Control-Defect_Detection

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Quick Start (Recommended)

**Option A - PowerShell:**
```powershell
.\start.ps1
```

**Option B - Batch:**
```bash
start.bat
```

This automatically launches:
- FastAPI backend at http://127.0.0.1:8000
- Streamlit frontend at http://localhost:8501

### Manual Start

**Terminal 1 - Backend API:**
```bash
.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

**Terminal 2 - Frontend UI:**
```bash
.venv\Scripts\Activate.ps1
streamlit run app.py
```

### Access Points

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://127.0.0.1:8000/docs |
| API Endpoint | http://127.0.0.1:8000/predict |

---

## Training

### Classification Model (Binary: PASS/DEFECT)

```bash
python train_model.py
```

**Configuration:**
- Base model: MobileNetV2 (ImageNet pretrained)
- Image size: 224×224
- Batch size: 32
- Epochs: 15 (head) + 5 (fine-tuning)
- Data augmentation: rotation, zoom, flip, brightness

**Output:** `visionspec_qc_model.keras`

### Detection Model (Multi-class YOLOv8)

```bash
python train_detection.py --epochs 50 --batch 16 --imgsz 640
```

**Configuration:**
- Model: YOLOv8n (nano)
- Data config: `data.yaml`
- Image size: 640×640
- Classes: 6 (open, short, mouse_bite, spur, pinhole, spurious)

**Output:** `runs/detect/pcb_yolov8n/weights/best.pt`

### Data Preparation

Visualize augmentation:
```bash
python train_data_preparation.py
```

---

## Models

### Classification Model
- **Architecture**: MobileNetV2 + Custom Classification Head
- **Input**: 224×224×3 RGB image
- **Output**: Probability (0=DEFECT, 1=PASS)
- **Threshold**: 0.7 (configurable)

### Detection Model
- **Architecture**: YOLOv8n
- **Input**: 640×640×3 RGB image
- **Output**: Bounding boxes with class labels and confidence scores
- **Classes**: 6 defect types

---

## API Documentation

### POST /predict

Upload an image for inspection.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@pcb_image.jpg"
```

**Response:**
```json
{
  "status": "DEFECT",
  "detections": [
    {
      "label": "open",
      "confidence": 0.95,
      "bbox": [120, 80, 250, 180]
    }
  ],
  "inference_time": "45.2ms"
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "PASS" or "DEFECT" |
| `detections` | array | List of detected defects |
| `detections[].label` | string | Defect class name |
| `detections[].confidence` | float | Confidence score (0-1) |
| `detections[].bbox` | array | [x1, y1, x2, y2] bounding box |
| `inference_time` | string | Processing time in milliseconds |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **Classification** | TensorFlow/Keras (MobileNetV2) |
| **Detection** | Ultralytics YOLOv8 |
| **Image Processing** | OpenCV, Pillow |
| **Data Processing** | NumPy, Matplotlib |

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- DeepPCB dataset for training data
- Ultralytics for YOLOv8
- TensorFlow/Keras for classification model