# YOLO Training Repository - Complete Usage Guide

Complete pipeline documentation for cell detection with YOLO models.

## 📑 Table of Contents

1. [Environment Setup](#-environment-setup)
2. [Dataset Conversion](#-dataset-conversion)
3. [Training Models](#-training-models)
4. [Inference & Evaluation](#-inference--evaluation)
5. [Monitoring & Visualization](#-monitoring--visualization)
6. [Complete Workflows](#-complete-workflows)
7. [Available Models](#-available-models)
8. [Troubleshooting](#-troubleshooting)

---

## 🔧 Environment Setup

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA-enabled GPU (recommended)
nvidia-smi
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install manually
pip install ultralytics torch torchvision opencv-python pandas numpy pyyaml

# Verify installation
python -c "from ultralytics import YOLO; print('✅ Ultralytics installed successfully')"
```

### Activate Environment

```bash
# If using conda
conda activate yolo_training

# If using venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

---

## 📦 Dataset Conversion

### Converting Cell Detection Dataset

Convert LabelMe+CSV format to YOLO format:

```bash
python scripts/convert_cell_dataset.py \
  --dataset-dir /path/to/source/dataset \
  --output-dir datasets_phx4 \
  --dataset-name cell_detection_phx4 \
  --class-names cell
```

**Expected Source Structure:**
```
source_dataset/
├── dataset.csv              # Contains: image_id, width, height, bbox, class, source
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
```

**Output Structure:**
```
datasets_phx4/
├── images/
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   └── test/               # Test images
├── labels/
│   ├── train/              # YOLO format labels
│   ├── val/
│   └── test/
└── cell_detection_phx4.yaml  # Dataset config
```

### Example: PHX4 Dataset

```bash
# Convert PHX4 dataset
python scripts/convert_cell_dataset.py \
  --dataset-dir /mnt/f/Datasets/celldetection/dataset_phx4 \
  --output-dir datasets_phx4 \
  --dataset-name cell_detection_phx4

# Verify conversion
ls -lh datasets_phx4/images/train/ | head
ls -lh datasets_phx4/labels/train/ | head
```

---

## 🚀 Training Models

### Single Model Training

#### Using Configuration Files (Recommended)

```bash
# Train with configuration file
python train_single_model.py --config configs/training/cell_detection_phx4.yaml
```

#### Available Training Configs

**PHX4 Dataset:**
- `configs/training/cell_detection_phx4.yaml` - YOLO11n (fast, 2.6M params)
- `configs/training/cell_detection_phx4_small.yaml` - YOLO11s (balanced, 9.4M params)
- `configs/training/cell_detection_phx4_medium.yaml` - YOLO11m (accurate, 25.9M params)

**Original Dataset:**
- `configs/training/cell_detection.json` - Standard training config

#### Training Parameters

Edit the YAML config file to customize:

```yaml
model: yolo11n.pt           # Model file
data: configs/datasets/cell_detection_phx4.yaml  # Dataset config
epochs: 100                 # Training epochs
batch_size: 16              # Batch size (reduce if OOM)
img_size: 1024              # Input image size
device: "0"                 # GPU device (0, 1, 2, etc. or "cpu")
patience: 20                # Early stopping patience
save_period: 10             # Save checkpoint every N epochs
cache: true                 # Cache images in RAM
workers: 8                  # Data loading workers
project: training_results_phx4  # Output directory
name: cell_detection_phx4_yolo11n  # Experiment name
```

### Multiple Model Training

```bash
# Train all models sequentially
python train_all_models.py --model all

# Train specific model family
python train_all_models.py --model yolov8
python train_all_models.py --model yolo11

# Train specific model
python train_all_models.py --model yolo11n
```

### Training Examples

#### Quick Test (Nano Model)
```bash
python train_single_model.py --config configs/training/cell_detection_phx4.yaml
```
- **Time:** ~2-4 hours
- **GPU Memory:** ~4GB
- **Best for:** Quick experiments, testing pipeline

#### Production Training (Small Model)
```bash
python train_single_model.py --config configs/training/cell_detection_phx4_small.yaml
```
- **Time:** ~4-6 hours
- **GPU Memory:** ~6GB
- **Best for:** Production deployment, balanced accuracy/speed

#### Maximum Accuracy (Medium Model)
```bash
python train_single_model.py --config configs/training/cell_detection_phx4_medium.yaml
```
- **Time:** ~6-8 hours
- **GPU Memory:** ~8GB
- **Best for:** Research, maximum accuracy requirements

### Training Output

```
training_results_phx4/
└── cell_detection_phx4_yolo11n/
    ├── weights/
    │   ├── best.pt         # Best model (highest mAP)
    │   └── last.pt         # Latest checkpoint
    ├── args.yaml           # Training arguments
    ├── results.csv         # Metrics per epoch
    ├── results.png         # Training curves
    ├── confusion_matrix.png
    ├── F1_curve.png
    ├── P_curve.png
    ├── PR_curve.png
    └── R_curve.png
```

---

## 🎯 Inference & Evaluation

### Basic Inference

```bash
# Inference on a single image
python scripts/inference.py \
  --model training_results_phx4/cell_detection_phx4_yolo11n/weights/best.pt \
  --image path/to/image.jpg

# Search for image by name in dataset directories
python scripts/inference.py \
  --model models/best.pt \
  --search Aequalized_Image_01.jpg

# Custom save directory
python scripts/inference.py \
  --model models/best.pt \
  --search test_image.jpg \
  --save-dir results/inference_output
```

### Inference Options

```bash
# Hide labels (show only bounding boxes)
python scripts/inference.py \
  --model models/best.pt \
  --search image.jpg \
  --label_switch

# Adjust confidence threshold
python scripts/inference.py \
  --model models/best.pt \
  --image image.jpg \
  --confidence 0.3

# Use CPU instead of GPU
python scripts/inference.py \
  --model models/best.pt \
  --image image.jpg \
  --device cpu
```

### Ground Truth Comparison (NEW!)

Compare predictions with ground truth using color-coded visualization:

```bash
python scripts/inference.py \
  --model models/best.pt \
  --search test_image.jpg \
  --compare-gt
```

**Color Scheme:**
- 🔵 **Blue**: False Negatives (ground truth missed by model)
- 🟢 **Green**: True Positives (correct predictions)
- 🔴 **Red**: False Positives (incorrect predictions)

**Advanced GT Comparison:**

```bash
# Custom IoU threshold for matching
python scripts/inference.py \
  --model models/best.pt \
  --search test_image.jpg \
  --compare-gt \
  --iou-threshold 0.6

# Hide labels (boxes only)
python scripts/inference.py \
  --model models/best.pt \
  --search test_image.jpg \
  --compare-gt \
  --label_switch

# Save to specific directory
python scripts/inference.py \
  --model models/best.pt \
  --search test_image.jpg \
  --compare-gt \
  --save-dir results/comparison
```

### Batch Evaluation

#### Validate on Test Set

```bash
# Run validation
python -c "
from ultralytics import YOLO
model = YOLO('training_results_phx4/cell_detection_phx4_yolo11n/weights/best.pt')
metrics = model.val(
    data='configs/datasets/cell_detection_phx4.yaml',
    split='test',
    conf=0.25
)
print(f'mAP50: {metrics.box.map50:.4f}')
print(f'mAP50-95: {metrics.box.map:.4f}')
print(f'Precision: {metrics.box.mp:.4f}')
print(f'Recall: {metrics.box.mr:.4f}')
"
```

#### Compare Multiple Models

```bash
# Evaluate all trained models
python evaluate_models.py --results-dir training_results_phx4

# Compare specific models
python compare_models.py \
  --models training_results_phx4/*/weights/best.pt
```

---

## 📊 Monitoring & Visualization

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir training_results_phx4/

# Access at http://localhost:6006
```

### Training Monitor

```bash
# Monitor training progress in real-time
python scripts/monitor.py \
  --project training_results_phx4/cell_detection_phx4_yolo11n
```

### View Results

```bash
# View training curves
xdg-open training_results_phx4/cell_detection_phx4_yolo11n/results.png

# View confusion matrix
xdg-open training_results_phx4/cell_detection_phx4_yolo11n/confusion_matrix.png

# Check metrics CSV
cat training_results_phx4/cell_detection_phx4_yolo11n/results.csv
```

---

## 🔄 Complete Workflows

### Workflow 1: Train & Evaluate New Dataset

```bash
# 1. Convert dataset
python scripts/convert_cell_dataset.py \
  --dataset-dir /path/to/dataset \
  --output-dir datasets_new \
  --dataset-name cell_detection_new

# 2. Create dataset config (or edit existing)
cat > configs/datasets/cell_detection_new.yaml << EOF
path: datasets_new
train: images/train
val: images/val
test: images/test
nc: 1
names: ['cell']
EOF

# 3. Create training config
cat > configs/training/cell_detection_new.yaml << EOF
model: yolo11n.pt
data: configs/datasets/cell_detection_new.yaml
epochs: 100
batch_size: 16
img_size: 1024
device: "0"
patience: 20
save_period: 10
cache: true
workers: 8
project: training_results_new
name: cell_detection_new_yolo11n
EOF

# 4. Train model
python train_single_model.py --config configs/training/cell_detection_new.yaml

# 5. Evaluate on test set
python scripts/inference.py \
  --model training_results_new/cell_detection_new_yolo11n/weights/best.pt \
  --search test_image.jpg \
  --compare-gt
```

### Workflow 2: Quick Model Testing

```bash
# 1. Train nano model (fast)
python train_single_model.py --config configs/training/cell_detection_phx4.yaml

# 2. Test inference
python scripts/inference.py \
  --model training_results_phx4/cell_detection_phx4_yolo11n/weights/best.pt \
  --search test_image.jpg \
  --compare-gt

# 3. If results good, train larger model
python train_single_model.py --config configs/training/cell_detection_phx4_small.yaml
```

### Workflow 3: Model Comparison

```bash
# 1. Train multiple models
python train_single_model.py --config configs/training/cell_detection_phx4.yaml
python train_single_model.py --config configs/training/cell_detection_phx4_small.yaml
python train_single_model.py --config configs/training/cell_detection_phx4_medium.yaml

# 2. Compare results
python evaluate_models.py --results-dir training_results_phx4

# 3. Test best model
python scripts/inference.py \
  --model training_results_phx4/cell_detection_phx4_yolo11m/weights/best.pt \
  --search test_image.jpg \
  --compare-gt
```

### Workflow 4: Hyperparameter Tuning

```bash
# 1. Create config with different parameters
cp configs/training/cell_detection_phx4.yaml configs/training/cell_detection_phx4_tune.yaml

# 2. Edit config (e.g., change learning rate, augmentation)
# ... edit YAML file ...

# 3. Train with new config
python train_single_model.py --config configs/training/cell_detection_phx4_tune.yaml

# 4. Compare with baseline
python compare_models.py \
  --models training_results_phx4/cell_detection_phx4_yolo11n/weights/best.pt \
           training_results_phx4/cell_detection_phx4_yolo11n_tune/weights/best.pt
```

---

## 📋 Available Models

### YOLO11 Models (Recommended)

| Model | Parameters | Speed | GPU Memory | Batch Size | Use Case |
|-------|-----------|-------|------------|------------|----------|
| yolo11n | 2.6M | Fast | ~4GB | 16 | Quick testing, edge devices |
| yolo11s | 9.4M | Medium | ~6GB | 16 | Production, balanced |
| yolo11m | 25.9M | Slow | ~8GB | 8 | Maximum accuracy |
| yolo11l | 44.2M | Slower | ~12GB | 4 | Research |
| yolo11x | 56.9M | Slowest | ~16GB | 4 | Benchmark |

### YOLOv8 Models

| Model | Parameters | Speed | GPU Memory | Batch Size |
|-------|-----------|-------|------------|------------|
| yolov8n | 3.2M | Fast | ~4GB | 16 |
| yolov8s | 11.2M | Medium | ~6GB | 12 |
| yolov8m | 25.9M | Slow | ~8GB | 8 |
| yolov8l | 43.7M | Slower | ~12GB | 6 |
| yolov8x | 68.2M | Slowest | ~16GB | 4 |

### Model Selection Guide

**Choose based on:**
- **Speed Priority**: yolo11n or yolo11s
- **Accuracy Priority**: yolo11m or yolo11l
- **Balanced**: yolo11s
- **Edge Deployment**: yolo11n
- **GPU Limited**: yolo11n (batch 16) or yolo11s (batch 8)

---

## 🛠️ Troubleshooting

### Out of Memory (OOM) Error

```bash
# Solution 1: Reduce batch size
# Edit config: batch_size: 8  # or 4

# Solution 2: Reduce image size
# Edit config: img_size: 640  # instead of 1024

# Solution 3: Disable cache
# Edit config: cache: false

# Solution 4: Use smaller model
python train_single_model.py --config configs/training/cell_detection_phx4.yaml  # nano instead of medium
```

### Slow Training

```bash
# Solution 1: Enable caching
# Edit config: cache: true

# Solution 2: Increase workers
# Edit config: workers: 8  # match CPU cores

# Solution 3: Use mixed precision
# Add to config: amp: true

# Solution 4: Reduce image size
# Edit config: img_size: 640
```

### Poor Validation Performance

```bash
# Solution 1: Increase epochs
# Edit config: epochs: 150

# Solution 2: Reduce early stopping patience
# Edit config: patience: 30

# Solution 3: Adjust learning rate
# Add to config: lr0: 0.001

# Solution 4: More data augmentation
# Add to config:
#   hsv_h: 0.015
#   hsv_s: 0.7
#   hsv_v: 0.4
```

### Model Not Found Error

```bash
# Download YOLO models manually
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo11m.pt

# Or let ultralytics download automatically on first run
```

### Inference Not Finding Labels

```bash
# Ensure label file exists in standard YOLO structure:
# datasets/images/val/image.jpg  ->  datasets/labels/val/image.txt

# Check with:
python scripts/inference.py --model models/best.pt --search image.jpg --compare-gt
# Should print: "Found GT labels: ..."
```

### Permission Errors on Windows (WSL)

```bash
# Fix file permissions
chmod +x scripts/*.py
chmod +x *.py

# Or run with python explicitly
python scripts/inference.py ...
```

---

## 📚 Additional Resources

### Documentation Files
- **`README.md`** - Repository overview
- **`PHX4_QUICK_START.md`** - PHX4 dataset quick reference
- **`docs/PHX4_TRAINING_GUIDE.md`** - Detailed PHX4 training guide
- **`USAGE.md`** - This file (complete usage guide)

### Configuration Files
- **`configs/datasets/`** - Dataset configurations
- **`configs/training/`** - Training configurations

### Scripts
- **`scripts/convert_cell_dataset.py`** - Dataset conversion
- **`scripts/inference.py`** - Inference with GT comparison
- **`scripts/monitor.py`** - Training monitoring
- **`train_single_model.py`** - Single model training
- **`train_all_models.py`** - Multiple model training
- **`evaluate_models.py`** - Model evaluation

---

## 🎯 Best Practices

1. **Start Small**: Begin with nano model for quick validation
2. **Monitor Early**: Check first 10 epochs to ensure training is working
3. **Use Caching**: Enable `cache: true` for faster training
4. **Save Regularly**: Keep `save_period: 10` to avoid losing progress
5. **Compare GT**: Always use `--compare-gt` to visualize model errors
6. **Test Multiple Sizes**: Try different image sizes (640, 800, 1024)
7. **Track Experiments**: Use meaningful experiment names
8. **Backup Models**: Keep best.pt files from different experiments

---

## 💡 Quick Reference

### Common Commands

```bash
# Training
python train_single_model.py --config configs/training/cell_detection_phx4.yaml

# Inference
python scripts/inference.py --model models/best.pt --search image.jpg --compare-gt

# Evaluation
python evaluate_models.py --results-dir training_results_phx4

# Monitoring
tensorboard --logdir training_results_phx4/
```

### Key Directories

```
├── configs/              # All configuration files
├── datasets_phx4/        # PHX4 dataset (YOLO format)
├── training_results_phx4/  # Training outputs
├── results/              # Inference results
├── models/               # Saved model files
└── scripts/              # Utility scripts
```

### Configuration Hierarchy

```
1. Dataset Config (YAML)  ->  configs/datasets/cell_detection_phx4.yaml
2. Training Config (YAML) ->  configs/training/cell_detection_phx4.yaml
3. Run Training           ->  python train_single_model.py --config ...
4. Model Output           ->  training_results_phx4/.../weights/best.pt
5. Run Inference          ->  python scripts/inference.py --model best.pt
```

---

**Need Help?** Check the specific documentation files for detailed guides or raise an issue in the repository.

**Last Updated:** October 2025
