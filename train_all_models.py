#!/usr/bin/env python3
"""
Comprehensive YOLO Training Script for Cell Detection
Trains all available YOLO models (v8, v9, v11) on the cell detection dataset.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config_path, model_name):
    """Train a single model using the specified configuration."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create model
    model = YOLO(config['model'])
    
    # Start training
    try:
        results = model.train(
            data=config['data'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['img_size'],
            device=config['device'],
            patience=config['patience'],
            save_period=config['save_period'],
            cache=config['cache'],
            workers=config['workers'],
            project=config['project'],
            name=config['name'],
            exist_ok=True
        )
        print(f"✅ Training completed for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Training failed for {model_name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train YOLO models for cell detection')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'yolov8', 'yolov9', 'yolo11', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 
                               'yolov9c', 'yolov9e', 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x', 'cpu_yolov8n'],
                       help='Model(s) to train')
    parser.add_argument('--config-dir', type=str, default='configs/models',
                       help='Directory containing model configurations')
    
    args = parser.parse_args()
    
    # Define model configurations
    model_configs = {
        'yolov8n': 'configs/models/yolov8n.yaml',
        'yolov8s': 'configs/models/yolov8s.yaml',
        'yolov8m': 'configs/models/yolov8m.yaml',
        'yolov8l': 'configs/models/yolov8l.yaml',
        'yolov8x': 'configs/models/yolov8x.yaml',
        'yolov9c': 'configs/models/yolov9c.yaml',
        'yolov9e': 'configs/models/yolov9e.yaml',
        'yolo11n': 'configs/models/yolo11n.yaml',
        'yolo11s': 'configs/models/yolo11s.yaml',
        'yolo11m': 'configs/models/yolo11m.yaml',
        'yolo11l': 'configs/models/yolo11l.yaml',
        'yolo11x': 'configs/models/yolo11x.yaml',
        'cpu_yolov8n': 'configs/models/cpu_yolov8n.yaml',
    }
    
    # Determine which models to train
    models_to_train = []
    
    if args.model == 'all':
        models_to_train = list(model_configs.keys())
    elif args.model == 'yolov8':
        models_to_train = [k for k in model_configs.keys() if k.startswith('yolov8')]
    elif args.model == 'yolov9':
        models_to_train = [k for k in model_configs.keys() if k.startswith('yolov9')]
    elif args.model == 'yolo11':
        models_to_train = [k for k in model_configs.keys() if k.startswith('yolo11')]
    else:
        if args.model in model_configs:
            models_to_train = [args.model]
        else:
            print(f"❌ Unknown model: {args.model}")
            sys.exit(1)
    
    print(f"🚀 Starting training for {len(models_to_train)} model(s): {', '.join(models_to_train)}")
    
    # Train each model
    successful_trains = 0
    for model_name in models_to_train:
        config_path = model_configs[model_name]
        
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            continue
            
        if train_model(config_path, model_name):
            successful_trains += 1
    
    print(f"\n{'='*60}")
    print(f"Training Summary:")
    print(f"✅ Successfully trained: {successful_trains}/{len(models_to_train)} models")
    print(f"❌ Failed: {len(models_to_train) - successful_trains} models")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 