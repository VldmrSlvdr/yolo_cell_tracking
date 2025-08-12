# YOLO Training Repository

A comprehensive repository for training YOLO8-11 models on custom datasets with easy configuration management.

## 🚀 Features

- **Multi-YOLO Support**: YOLO8, YOLO9, YOLO10, YOLO11
- **Easy Configuration**: JSON-based config files
- **Dataset Conversion**: COCO to YOLO format conversion
- **Training Monitoring**: Real-time progress tracking
- **Inference Tools**: Easy model testing and evaluation
- **Model Management**: Save, load, and compare models

## 📁 Repository Structure

```
yolo_training_repo/
├── configs/                 # Configuration files
│   ├── models/             # Model configurations
│   ├── datasets/           # Dataset configurations
│   └── training/           # Training configurations
├── scripts/
│   ├── convert_dataset.py  # Dataset conversion
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   ├── monitor.py         # Training monitoring
│   └── utils.py           # Utility functions
├── datasets/              # Your datasets
    ├──images/
        ├──train/
           ├──image_train_001.jpg
           ├──... 
        ├──val/
           ├──image_val_001.jpg
           ├──...
    ├──labels/
        ├──train/
           ├──image_train_001.txt
           ├──... 
        ├──val/             
           ├──image_val_001.txt
           ├──... 
    ├──cell_detection.yaml  # Should follow yolo dataset format
├── models/                 # Trained models
├── results/                # Training results
└── examples/              # Example configurations
```

## 🛠️ Quick Start

### 1. Setup Environment

```bash
# Install requirements
pip install ultralytics torch torchvision

# Or create conda environment
conda create -n yolo_training python=3.10
conda activate yolo_training
pip install ultralytics torch torchvision
```

### 2. Prepare Dataset

```bash
# Convert COCO format to YOLO format
python scripts/convert_dataset.py --config configs/datasets/my_dataset.json

# Or use the interactive converter
python scripts/convert_dataset.py --interactive
```

### 3. Train Model

```bash
# Train with default config
python scripts/train.py --config configs/training/default.json

# Train with custom config
python scripts/train.py --config configs/training/custom.json

# Train specific model
python scripts/train.py --model yolo11n --epochs 100 --batch-size 16
```

### 4. Run Inference

```bash
# Basic inference
python scripts/inference.py --model models/best.pt --image path/to/image.jpg

# Search by image name in dataset directories
python scripts/inference.py --model models/best.pt --search image_name

# Hide labels in output visualization (show only bounding boxes)
python scripts/inference.py --model models/best.pt --search image_name --label_switch

# Save results to custom directory
python scripts/inference.py --model models/best.pt --image path/to/image.jpg --save-dir results/
```

## 📋 Configuration Management

### Model Configurations

```json
// configs/models/yolo11n.json
{
  "name": "yolo11n",
  "model": "yolo11n.pt",
  "size": "nano",
  "parameters": "2.6M",
  "description": "Fastest YOLO11 model"
}
```

### Dataset Configurations

```json
// configs/datasets/cell_detection.json
{
  "name": "cell_detection",
  "path": "datasets/cell_detection",
  "train": "images/train",
  "val": "images/val",
  "nc": 1,
  "names": ["cell"],
  "description": "Cell detection dataset"
}
```

### Training Configurations

```json
// configs/training/default.json
{
  "model": "yolo11n.pt",
  "data": "configs/datasets/cell_detection.json",
  "epochs": 100,
  "batch_size": 16,
  "img_size": 640,
  "device": "0",
  "patience": 20,
  "save_period": 10,
  "cache": true,
  "workers": 8,
  "project": "training_results",
  "name": "exp"
}
```

## 🔧 Advanced Usage

### Custom Training Configuration

```bash
# Create custom config
python scripts/create_config.py --template training --output my_config.json

# Edit config
python scripts/edit_config.py --config my_config.json

# Validate config
python scripts/validate_config.py --config my_config.json
```

### Model Comparison

```bash
# Compare multiple models
python scripts/compare_models.py --models models/model1.pt models/model2.pt

# Benchmark models
python scripts/benchmark.py --models yolo8n yolo9n yolo10n yolo11n
```

### Training Monitoring

```bash
# Monitor training in real-time
python scripts/monitor.py --project training_results/exp

# Generate training report
python scripts/report.py --project training_results/exp
```

## 📊 Supported Models

| Model | Version | Parameters | Speed | Accuracy |
|-------|---------|------------|-------|----------|
| YOLO8n | 8.x | 3.2M | Fast | Good |
| YOLO8s | 8.x | 11.2M | Medium | Better |
| YOLO8m | 8.x | 25.9M | Slow | Best |
| YOLO9n | 9.x | 2.6M | Fast | Good |
| YOLO9s | 9.x | 9.4M | Medium | Better |
| YOLO10n | 10.x | 2.6M | Fast | Good |
| YOLO10s | 10.x | 9.4M | Medium | Better |
| YOLO11n | 11.x | 2.6M | Fast | Good |
| YOLO11s | 11.x | 9.4M | Medium | Better |

## 🎯 Examples

### Cell Detection

```bash
# Convert cell dataset
python scripts/convert_dataset.py --config configs/datasets/cell_detection.json

# Train YOLO11n for cell detection
python scripts/train.py --config configs/training/cell_detection.json

# Test inference with labels hidden
python scripts/inference.py --model models/best.pt --search cell_image --label_switch
```

### Object Detection

```bash
# Train on custom dataset
python scripts/train.py --model yolo11s --data my_dataset.yaml --epochs 200

# Multi-class detection
python scripts/train.py --config configs/training/multi_class.json
```

## 📈 Monitoring and Visualization

### Training Progress

```bash
# Real-time monitoring
python scripts/monitor.py --project training_results/exp

# Generate plots
python scripts/plot_results.py --project training_results/exp
```

### Model Evaluation

```bash
# Validate model
python scripts/validate.py --model models/best.pt --data configs/datasets/my_dataset.json

# Generate confusion matrix
python scripts/confusion_matrix.py --model models/best.pt
```

## 🎯 Inference Script Features

The inference script (`scripts/inference.py`) provides flexible options for running predictions:

### Command Line Arguments

```bash
# Required
--model MODEL                    # Path to trained YOLO model

# Input options (choose one)
--image IMAGE                    # Direct path to input image
--search IMAGE_NAME              # Search for image by name in dataset directories

# Output options
--save-dir DIRECTORY             # Custom output directory (default: runs/detect/predict)
--label_switch                   # Hide labels/confidence in visualization (show only boxes)

# Detection options
--confidence THRESHOLD           # Confidence threshold (default: 0.25)
--device DEVICE                  # Device to use: 0 for GPU, cpu for CPU (default: 0)
```

### Output Files

The script generates:
- **Visualization image**: Shows detections with/without labels based on `--label_switch`
- **Label files**: Text files with bounding box coordinates (always saved)
- **Organized structure**: Both saved to the same directory

### Examples

```bash
# Basic inference with labels shown
python scripts/inference.py --model training_results/cell_detection_yolo11s/weights/best.pt --search cell_image

# Clean visualization (boxes only, no text labels)
python scripts/inference.py --model training_results/cell_detection_yolo11s/weights/best.pt --search cell_image --label_switch

# Custom output location
python scripts/inference.py --model models/best.pt --image data/test.jpg --save-dir my_results/ --label_switch

# Lower confidence threshold
python scripts/inference.py --model models/best.pt --search test_image --confidence 0.1
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/train.py --batch-size 8
   
   # Use smaller model
   python scripts/train.py --model yolo11n
   ```

2. **Slow Training**
   ```bash
   # Enable caching
   python scripts/train.py --cache
   
   # Reduce image size
   python scripts/train.py --img-size 512
   ```

3. **Poor Detection**
   ```bash
   # Increase epochs
   python scripts/train.py --epochs 200
   
   # Use larger model
   python scripts/train.py --model yolo11s
   ```

## 📚 Documentation

- [Configuration Guide](docs/configuration.md)
- [Training Guide](docs/training.md)
- [Inference Guide](docs/inference.md)
- [Model Comparison](docs/models.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details. 