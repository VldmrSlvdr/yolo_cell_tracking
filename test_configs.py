#!/usr/bin/env python3
"""
Test Configuration Script
Validates all model configurations and dataset setup.
"""

import os
import sys
import yaml
from pathlib import Path

def test_config(config_path, config_name):
    """Test a single configuration file."""
    print(f"Testing {config_name}...")
    
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
        
        # Load and validate YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['model', 'data', 'epochs', 'batch_size', 'img_size']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check if data file exists
        if not os.path.exists(config['data']):
            print(f"❌ Data configuration not found: {config['data']}")
            return False
        
        print(f"✅ {config_name} - Valid")
        return True
        
    except Exception as e:
        print(f"❌ {config_name} - Error: {str(e)}")
        return False

def test_dataset():
    """Test dataset configuration."""
    print("Testing dataset configuration...")
    
    dataset_config = "datasets/cell_detection.yaml"
    if not os.path.exists(dataset_config):
        print(f"❌ Dataset configuration not found: {dataset_config}")
        return False
    
    try:
        with open(dataset_config, 'r') as f:
            dataset = yaml.safe_load(f)
        
        # Check dataset structure
        train_dir = os.path.join(dataset['path'], dataset['train'])
        val_dir = os.path.join(dataset['path'], dataset['val'])
        
        if not os.path.exists(train_dir):
            print(f"❌ Training images not found: {train_dir}")
            return False
        
        if not os.path.exists(val_dir):
            print(f"❌ Validation images not found: {val_dir}")
            return False
        
        print(f"✅ Dataset configuration - Valid")
        return True
        
    except Exception as e:
        print(f"❌ Dataset configuration - Error: {str(e)}")
        return False

def main():
    print("🔍 Testing YOLO Training Configurations")
    print("=" * 50)
    
    # Test dataset
    dataset_ok = test_dataset()
    print()
    
    # Test model configurations
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
    }
    
    config_results = []
    for model_name, config_path in model_configs.items():
        result = test_config(config_path, model_name)
        config_results.append(result)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Dataset: {'✅ Valid' if dataset_ok else '❌ Invalid'}")
    print(f"Model Configurations: {sum(config_results)}/{len(config_results)} valid")
    
    if dataset_ok and all(config_results):
        print("\n🎉 All configurations are valid! Ready for training.")
    else:
        print("\n⚠️  Some configurations need attention before training.")

if __name__ == "__main__":
    main() 