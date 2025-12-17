# Polyp Segmentation with Knowledge Distillation

A deep learning project for **polyp segmentation** in colonoscopy images using **Knowledge Distillation**. This project trains a lightweight MobileNetV2 student model to achieve performance close to a larger ResNet50 teacher model while maintaining fast inference speed.

## Table of Contents

- [Overview](#Overview)
- [Features](#Features)
- [Project Structure](#Project-structure)
- [Installation](#Installation)
- [Usage](#Usage)
- [Training Modes](#Training-modes)
- [Model Architecture](#Model-architecture)
- [Dataset](#Dataset)
- [Results](#Results)

## Overview

Polyps are abnormal growths on the inner lining of the colon that can potentially develop into colorectal cancer. Early detection and removal of polyps during colonoscopy is crucial for cancer prevention. This project provides an automated polyp segmentation system that can assist medical professionals in identifying polyp regions.

### Key Highlights

- **Knowledge Distillation**: Transfer knowledge from a large teacher model to a compact student model
- **Lightweight Architecture**: MobileNetV2 backbone for real-time inference
- **High Accuracy**: Achieves competitive Dice scores on standard polyp benchmarks
- **Easy Deployment**: Includes Gradio web interface for easy demo and testing

## Features

- **Three Training Modes**: Baseline, Knowledge Distillation, and Teacher training
- **Comprehensive Evaluation**: Test on 5 public polyp datasets
- **Visual Results**: Save prediction visualizations
- **Web Demo**: Interactive Gradio application
- **Fast Inference**: Optimized for real-time segmentation

## Project Structure

```
├── main.py                 # Unified entry point (train/eval/app)
├── app.py                  # Gradio web application
├── requirements.txt        # Python dependencies
├── configs/
│   └── config.yaml         # Training configuration
├── src/
│   ├── train.py            # Unified training script
│   ├── evaluate.py         # Evaluation script
│   ├── plot_training_history.py
│   ├── models/
│   │   ├── student.py      # MobileNetV2 U-Net student model
│   │   ├── teacher.py      # Teacher model definition
│   │   └── teacher_smp.py  # SMP-based teacher model
│   └── utils/
│       ├── data_loader.py  # Dataset and data loading
│       ├── losses.py       # Loss functions (Dice, BCE, KD)
│       └── metrics.py      # Evaluation metrics
├── checkpoints/            # Saved model weights
├── dataset/
│   ├── TrainDataset/       # Training images and masks
│   └── TestDataset/        # Test datasets
│       ├── CVC-300/
│       ├── CVC-ClinicDB/
│       ├── CVC-ColonDB/
│       ├── ETIS-LaribPolypDB/
│       └── Kvasir/
└── results/
    └── visuals/            # Saved prediction visualizations
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download datasets**
   - Place training data in `dataset/TrainDataset/`
   - Place test data in `dataset/TestDataset/`

## Usage

The project uses a unified `main.py` entry point for all operations.

### Quick Start

```bash
# Launch Gradio demo (default)
python main.py

# Or explicitly
python main.py app
```

### Training

```bash
# Train with Knowledge Distillation (default)
python main.py train --mode distillation

# Train baseline model (no teacher)
python main.py train --mode baseline

# Train teacher model from scratch
python main.py train --mode teacher

# Use custom config
python main.py train --mode distillation --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate a trained model
python main.py eval --checkpoint checkpoints/MobileNetV2_Distillation_v1_best.pth

# Evaluate and save visual predictions
python main.py eval --checkpoint checkpoints/model.pth --save_visuals
```

### Plot Training History

```bash
# Plot training history from all checkpoint JSON files
python main.py plot

# Custom output path
python main.py plot --output results/my_comparison.png

# Use specific history files
python main.py plot --history_files checkpoints/Best_Baseline_history.json checkpoints/Best_kd_history.json

# Custom checkpoints directory
python main.py plot --checkpoints_dir my_checkpoints
```

### Gradio App

```bash
# Launch on default port (7860)
python main.py app

# Launch with public URL (shareable)
python main.py app --share

# Custom host and port
python main.py app --host 127.0.0.1 --port 8080
```

## Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `baseline` | Train student model from scratch with ground truth only | Baseline comparison |
| `distillation` | Train student with KD loss from pre-trained teacher | **Recommended** - Best performance |
| `teacher` | Train the teacher model (ResNet50) from scratch | Prepare teacher for distillation |

### Knowledge Distillation Formula

```
Total Loss = α × Supervised Loss + (1 - α) × KD Loss
```

Where:
- **Supervised Loss**: Dice + BCE loss with ground truth masks
- **KD Loss**: MSE loss between student and teacher logits (with temperature scaling)
- **α**: Weighting factor (default: 0.6)

## Model Architecture

### Student Model (MobileNetV2)
- **Encoder**: MobileNetV2 (pretrained on ImageNet)
- **Decoder**: U-Net style decoder with skip connections
- **Parameters**: ~4.5M
- **Input Size**: 320×320

### Teacher Model (ResNet50)
- **Encoder**: ResNet50 (pretrained on ImageNet)
- **Decoder**: U-Net++ decoder
- **Parameters**: ~32M
- **Input Size**: 320×320

## Dataset

The project uses standard polyp segmentation benchmarks:

| Dataset           | Images    | Description                |
|-------------------|-----------|----------------------------|
| Kvasir-SEG        | 100       | Training + Testing         |
| CVC-ClinicDB      | 62        | Validation during training |
| CVC-ColonDB       | 380       | Testing                    |
| CVC-300           | 60        | Testing                    |
| ETIS-LaribPolypDB | 196       | Testing                    |

### Dataset Structure

```
dataset/
├── TrainDataset/
│   ├── images/
│   │   ├── image1.png
│   │   └── ...
│   └── masks/
│       ├── image1.png
│       └── ...
└── TestDataset/
    └── <DatasetName>/
        ├── images/
        └── masks/
```

## Results

### Performance Comparison

| Model              | Kvasir | CVC-ClinicDB | CVC-ColonDB | CVC-300 |  ETIS  | Avg Dice |
|--------------------|--------|--------------|-------------|---------|--------|----------|
| Teacher            | 0.8776 | 0.9215       | 0.6765      | 0.8679  | 0.6378 | 0.7962   |
| Student (Baseline) | 0.8774 | 0.8810       | 0.6867      | 0.8500  | 0.6157 | 0.7822   |
| Student (KD)       | 0.8917 | 0.8915       | 0.6854      | 0.8380  | 0.6427 | 0.7898   |

## Gradio Demo

The project includes an interactive web demo built with Gradio.

### Features
- Upload colonoscopy images
- Real-time polyp segmentation
- Visualization with overlay and contours
- Example images from test datasets

### Screenshots

Launch the demo:
```bash
python main.py app
```

Then open `http://localhost:7860` in your browser.

## Configuration

Edit `configs/config.yaml` to customize training:

```yaml
# Experiment
experiment_name: "MobileNetV2_Distillation_v1"

# Paths
data_root: "dataset"
save_dir: "checkpoints"

# Model
student_backbone: "mobilenet_v2"
teacher_checkpoint: "checkpoints/teacher_supervised_best.pth"

# Training
img_size: 320
batch_size: 8
epochs: 50
learning_rate: 1.0e-4

# Knowledge Distillation
alpha: 0.6          # Weight for supervised loss
temperature: 4.0    # KD temperature
```

## Requirements

Main dependencies:
- `torch >= 2.0`
- `segmentation-models-pytorch`
- `albumentations`
- `gradio`
- `opencv-python`
- `pyyaml`
- `tqdm`

See `requirements.txt` for full list.