# YOLO Training Usage Guide

This repository contains comprehensive YOLO model configurations and training scripts for cell detection.

## Available Models

### YOLOv8 Models
- **yolov8n.yaml** - Nano (fastest, smallest)
- **yolov8s.yaml** - Small
- **yolov8m.yaml** - Medium
- **yolov8l.yaml** - Large
- **yolov8x.yaml** - Extra Large (most accurate)

### YOLOv9 Models
- **yolov9c.yaml** - Compact
- **yolov9e.yaml** - Efficient

### YOLO11 Models
- **yolo11n.yaml** - Nano
- **yolo11s.yaml** - Small
- **yolo11m.yaml** - Medium
- **yolo11l.yaml** - Large
- **yolo11x.yaml** - Extra Large

## Training Scripts

### 1. Train Single Model
```bash
python train_single_model.py --config configs/models/yolov8n.yaml
```

### 2. Train All Models
```bash
# Train all models
python train_all_models.py --model all

# Train specific model family
python train_all_models.py --model yolov8
python train_all_models.py --model yolov9
python train_all_models.py --model yolo11

# Train specific model
python train_all_models.py --model yolov8n
```

### 3. Evaluate Models
```bash
python evaluate_models.py --results-dir training_results
```

## Configuration Details

Each model configuration includes:
- **model**: Pre-trained model file
- **data**: Dataset configuration path
- **epochs**: Number of training epochs (100)
- **batch_size**: Adjusted for model size
- **img_size**: Input image size (1024)
- **device**: GPU device (0)
- **patience**: Early stopping patience (20)
- **save_period**: Save checkpoint every N epochs (10)
- **cache**: Cache images for faster training
- **workers**: Number of data loading workers (8)

## Batch Size Recommendations

- **Small models** (nano, small): 16-12 batch size
- **Medium models**: 8 batch size
- **Large models**: 6 batch size
- **Extra large models**: 4 batch size

## Expected Training Times

- **Nano models**: ~2-4 hours
- **Small models**: ~4-6 hours
- **Medium models**: ~6-8 hours
- **Large models**: ~8-12 hours
- **Extra large models**: ~12-16 hours

## Results

Training results will be saved in:
- `training_results/cell_detection_[model_name]/`
- Best model: `weights/best.pt`
- Last model: `weights/last.pt`
- Training logs and plots

## Model Comparison

After training, use the evaluation script to compare all models:
```bash
python evaluate_models.py
```

This will generate a CSV file with performance metrics for all trained models. 