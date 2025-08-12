#!/usr/bin/env python3
"""
Single Model Training Script for Cell Detection
Trains a single YOLO model using configuration files.
"""

import os
import sys
import yaml
import argparse
from ultralytics import YOLO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config_path):
    """Train a model using the specified configuration."""
    print(f"Loading configuration from: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    print(f"Model: {config['model']}")
    print(f"Data: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Image size: {config['img_size']}")
    print(f"Device: {config['device']}")
    
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
        print(f"✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train a single YOLO model for cell detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the model configuration file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    print(f"🚀 Starting training with configuration: {args.config}")
    train_model(args.config)

if __name__ == "__main__":
    main() 