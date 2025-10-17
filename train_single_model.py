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
    print(f"Batch size: {config.get('batch_size')}")
    print(f"Image size: {config.get('img_size')}")
    print(f"Device: {config['device']}")
    if 'max_det' in config:
        print(f"Max detections: {config['max_det']}")
    
    # Create model
    model = YOLO(config['model'])
    
    # Prepare training parameters - map config keys to YOLO parameter names
    param_mapping = {
        'batch_size': 'batch',
        'img_size': 'imgsz',
        'description': None,  # Skip description field
    }
    
    # Build training kwargs dynamically from ALL config parameters
    train_kwargs = {}
    for key, value in config.items():
        # Skip None values, description, and model (already used)
        if value is None or key in ['description', 'model']:
            continue
            
        # Get the YOLO parameter name (use mapping or original key)
        param_name = param_mapping.get(key, key)
        if param_name is None:
            continue
            
        train_kwargs[param_name] = value
    
    # Set exist_ok to False by default to prevent resuming with old parameters
    if 'exist_ok' not in train_kwargs:
        train_kwargs['exist_ok'] = False
    
    # Ensure save is enabled
    if 'save' not in train_kwargs:
        train_kwargs['save'] = True
    
    print(f"\n🔧 Training parameters:")
    print(f"   Total parameters: {len(train_kwargs)}")
    if 'max_det' in train_kwargs:
        print(f"   max_det: {train_kwargs['max_det']}")
    if 'exist_ok' in train_kwargs:
        print(f"   exist_ok: {train_kwargs['exist_ok']}")
    print()
    
    # Start training
    try:
        results = model.train(**train_kwargs)
        print(f"✅ Training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
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