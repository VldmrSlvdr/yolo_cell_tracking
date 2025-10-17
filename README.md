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

# Compare predictions with ground truth (blue=GT missed, green=correct, red=wrong)
python scripts/inference.py --model models/best.pt --search image_name --compare-gt

# Save results to custom directory
python scripts/inference.py --model models/best.pt --image path/to/image.jpg --save-dir results/
```

## 🧬 PHX4 Dataset Training

### Quick Start with PHX4 Dataset

```bash
# 1. Convert PHX4 dataset (if not already done)
python scripts/convert_cell_dataset.py \
  --dataset-dir /mnt/f/Datasets/celldetection/dataset_phx4 \
  --output-dir datasets_phx4 \
  --dataset-name cell_detection_phx4

# 2. Train with data augmentation (recommended)
python train_single_model.py --config configs/training/cell_detection_phx4.yaml

# 3. Train WITHOUT data augmentation (for comparison)
python train_single_model.py --config configs/training/cell_detection_phx4_no_aug.yaml

# 4. Train with small model (better accuracy)
python train_single_model.py --config configs/training/cell_detection_phx4_small.yaml

# 5. Train with medium model (best accuracy)
python train_single_model.py --config configs/training/cell_detection_phx4_medium.yaml

# 6. Run inference - max_det is automatically loaded from model's training config
python scripts/inference.py \
  --model training_results_phx4/cell_detection_phx4_yolov8n_16bs_no_dataaug_1500maxdet/weights/best.pt \
  --search test_image.jpg \
  --compare-gt \
  --save-dir results/phx4_inference
```

📖 **See [PHX4 Training Guide](docs/PHX4_TRAINING_GUIDE.md) for detailed instructions**

📋 **See [CHANGELOG.md](CHANGELOG.md) for recent updates and fixes**

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

YAML format is recommended for better readability and support for comments:

```yaml
# configs/training/cell_detection_phx4.yaml
model: yolov8n.pt
data: configs/datasets/cell_detection_phx4.yaml
epochs: 100
batch_size: 16
img_size: 1024
device: "0"
patience: 20
save_period: 10
cache: true
workers: 8
max_det: 1500  # Maximum detections per image (default: 300)
project: training_results_phx4
name: cell_detection_phx4_yolov8n_16bs_w_dataaug_1500maxdet

# Data Augmentation Settings (set to 0 to disable)
hsv_h: 0.015      # HSV-Hue augmentation
hsv_s: 0.7        # HSV-Saturation augmentation
hsv_v: 0.4        # HSV-Value augmentation
degrees: 0.0      # Image rotation
translate: 0.1    # Image translation
scale: 0.5        # Image scale
fliplr: 0.5       # Horizontal flip probability
mosaic: 1.0       # Mosaic augmentation
auto_augment: randaugment
erasing: 0.4      # Random erasing
```

**Key Parameters:**
- `max_det`: Maximum number of detections per image (increase for dense scenes)
- Data augmentation parameters can be disabled by setting to 0
- Use unique `name` for each experiment to avoid overwriting results

## ⚙️ Important Configuration Notes

### Maximum Detections (`max_det`)

The `max_det` parameter controls the maximum number of objects detected per image:
- Default: 300 detections per image
- For dense scenes (many cells): Set to 1500 or higher
- **Automatically applied**: Inference script reads `max_det` from trained model's config
- Manual override: Use `--max-det` flag during inference

```bash
# Training automatically saves max_det in model config
python train_single_model.py --config configs/training/cell_detection_phx4.yaml

# Inference automatically uses model's max_det (no flag needed!)
python scripts/inference.py --model path/to/best.pt --image test.jpg

# Or manually override if needed
python scripts/inference.py --model path/to/best.pt --image test.jpg --max-det 2000
```

### Data Augmentation Control

Toggle augmentation by editing config or using different config files:

```yaml
# Enable augmentation (configs/training/cell_detection_phx4.yaml)
hsv_h: 0.015
mosaic: 1.0
fliplr: 0.5

# Disable augmentation (configs/training/cell_detection_phx4_no_aug.yaml)
hsv_h: 0.0
mosaic: 0.0
fliplr: 0.0
```

### Best Practices

1. **Use unique experiment names** to avoid overwriting results
2. **Delete old training directories** before retraining with new parameters
3. **Verify training parameters** by checking `args.yaml` after training starts:
   ```bash
   cat training_results_phx4/your_experiment/args.yaml | grep max_det
   ```
4. **YAML configs recommended** over JSON for better readability

## 🔧 Advanced Usage

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

| Model | Version | Parameters | Speed | Accuracy | Recommended Use |
|-------|---------|------------|-------|----------|-----------------|
| YOLO8n | 8.x | 3.2M | Fast | Good | Testing, Fast inference |
| YOLO8s | 8.x | 11.2M | Medium | Better | Balanced performance |
| YOLO8m | 8.x | 25.9M | Slow | Best | High accuracy needed |
| YOLO11n | 11.x | 2.6M | Fast | Good | **Default choice** |
| YOLO11s | 11.x | 9.4M | Medium | Better | Production use |
| YOLO11m | 11.x | 20M+ | Slow | Best | Maximum accuracy |

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

## 📁 Repository Organization

### Active Files
- `train_single_model.py` - **Main training script** (use this!)
- `scripts/` - Inference, conversion, and utility scripts
- `configs/` - Training, model, and dataset configurations
- `docs/` - Documentation and guides

### Archived Files
- `archive/old_scripts/` - Deprecated scripts kept for reference
- `docs/archive/` - Resolved issue documentation

See `archive/README.md` and `docs/archive/README.md` for details.

## 📝 Recent Updates

See [CHANGELOG.md](CHANGELOG.md) for recent fixes and improvements:
- Fixed `max_det` parameter handling in training and inference
- Added automatic parameter reading from model configs
- Improved data augmentation controls
- Updated to YAML-based configuration

## 📄 License

MIT License - see LICENSE file for details. 