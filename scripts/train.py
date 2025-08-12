#!/usr/bin/env python3
"""
YOLO Training Script - Supports YOLO8-11 models
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate training configuration"""
    required_keys = ['model', 'data']
    optional_keys = ['epochs', 'batch_size', 'img_size', 'device', 'patience', 
                    'save_period', 'cache', 'workers', 'project', 'name']
    
    # Check required keys
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required config key: {key}")
            return False
    
    # Set defaults for optional keys
    defaults = {
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'device': '0',
        'patience': 20,
        'save_period': 10,
        'cache': True,
        'workers': 8,
        'project': 'training_results',
        'name': 'exp'
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    return True

def get_yolo_version(model_name: str) -> str:
    """Determine YOLO version from model name"""
    if model_name.startswith('yolo8'):
        return '8'
    elif model_name.startswith('yolo9'):
        return '9'
    elif model_name.startswith('yolo10'):
        return '10'
    elif model_name.startswith('yolo11'):
        return '11'
    else:
        return '11'  # Default to latest

def install_yolo(version: str = '11'):
    """Install specific YOLO version if needed"""
    try:
        import ultralytics
        print(f"✅ YOLO{version} already installed")
        return True
    except ImportError:
        print(f"📦 Installing YOLO{version}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ YOLO{version} installed successfully")
            return True
        else:
            print(f"❌ Failed to install YOLO{version}: {result.stderr}")
            return False

def run_training(config: Dict[str, Any], verbose: bool = True) -> bool:
    """Run YOLO training with configuration"""
    
    print(f"🚀 Starting YOLO training...")
    print(f"🤖 Model: {config['model']}")
    print(f"📁 Dataset: {config['data']}")
    print(f"⏱️  Epochs: {config['epochs']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🖼️  Image size: {config['img_size']}")
    print(f"💻 Device: {config['device']}")
    print(f"📊 Project: {config['project']}")
    print("=" * 60)
    
    # Build training command
    cmd = [
        "yolo", "train",
        f"model={config['model']}",
        f"data={config['data']}",
        f"epochs={config['epochs']}",
        f"batch={config['batch_size']}",
        f"imgsz={config['img_size']}",
        f"device={config['device']}",
        f"patience={config['patience']}",
        "save=True",
        f"save_period={config['save_period']}",
        f"cache={config['cache']}",
        f"workers={config['workers']}",
        f"project={config['project']}",
        f"name={config['name']}",
        "exist_ok=True"
    ]
    
    if verbose:
        print(f"🔧 Command: {' '.join(cmd)}")
        print("=" * 60)
    
    # Start training process
    if verbose:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor output in real-time
        print("📊 Training Progress:")
        print("-" * 40)
        
        for line in process.stdout:
            line = line.strip()
            
            # Filter and display important information
            if any(keyword in line.lower() for keyword in [
                "epoch", "loss", "map", "train", "val", "saving", "completed"
            ]):
                print(f"🔄 {line}")
            
            # Show progress indicators
            if "epoch" in line.lower() and "loss" in line.lower():
                print(f"📈 {line}")
            
            # Show completion status
            if "saving" in line.lower():
                print(f"💾 {line}")
            
            # Show final results
            if "completed" in line.lower():
                print(f"✅ {line}")
        
        # Wait for process to complete
        return_code = process.wait()
    else:
        # Run without verbose output
        result = subprocess.run(cmd, capture_output=True, text=True)
        return_code = result.returncode
    
    if return_code == 0:
        print("\n🎉 Training completed successfully!")
        return True
    else:
        print(f"\n❌ Training failed with return code: {return_code}")
        return False

def create_default_configs():
    """Create default configuration files"""
    
    # Model configs
    models = {
        'yolo8n': {'name': 'yolo8n', 'model': 'yolo8n.pt', 'size': 'nano', 'parameters': '3.2M'},
        'yolo8s': {'name': 'yolo8s', 'model': 'yolo8s.pt', 'size': 'small', 'parameters': '11.2M'},
        'yolo8m': {'name': 'yolo8m', 'model': 'yolo8m.pt', 'size': 'medium', 'parameters': '25.9M'},
        'yolo9n': {'name': 'yolo9n', 'model': 'yolo9n.pt', 'size': 'nano', 'parameters': '2.6M'},
        'yolo9s': {'name': 'yolo9s', 'model': 'yolo9s.pt', 'size': 'small', 'parameters': '9.4M'},
        'yolo10n': {'name': 'yolo10n', 'model': 'yolo10n.pt', 'size': 'nano', 'parameters': '2.6M'},
        'yolo10s': {'name': 'yolo10s', 'model': 'yolo10s.pt', 'size': 'small', 'parameters': '9.4M'},
        'yolo11n': {'name': 'yolo11n', 'model': 'yolo11n.pt', 'size': 'nano', 'parameters': '2.6M'},
        'yolo11s': {'name': 'yolo11s', 'model': 'yolo11s.pt', 'size': 'small', 'parameters': '9.4M'}
    }
    
    for model_name, model_config in models.items():
        config_path = f"configs/models/{model_name}.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
    
    # Training configs
    training_configs = {
        'default': {
            'model': 'yolo11n.pt',
            'data': 'configs/datasets/my_dataset.json',
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'device': '0',
            'patience': 20,
            'save_period': 10,
            'cache': True,
            'workers': 8,
            'project': 'training_results',
            'name': 'exp'
        },
        'fast': {
            'model': 'yolo11n.pt',
            'data': 'configs/datasets/my_dataset.json',
            'epochs': 50,
            'batch_size': 8,
            'img_size': 512,
            'device': '0',
            'patience': 10,
            'save_period': 5,
            'cache': True,
            'workers': 4,
            'project': 'training_results',
            'name': 'fast'
        },
        'accurate': {
            'model': 'yolo11s.pt',
            'data': 'configs/datasets/my_dataset.json',
            'epochs': 200,
            'batch_size': 16,
            'img_size': 1024,
            'device': '0',
            'patience': 30,
            'save_period': 20,
            'cache': True,
            'workers': 8,
            'project': 'training_results',
            'name': 'accurate'
        }
    }
    
    for config_name, config in training_configs.items():
        config_path = f"configs/training/{config_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to training configuration file")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (e.g., yolo11n, yolo8s)")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to dataset configuration")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                       help="Device to use (0 for GPU, cpu for CPU)")
    parser.add_argument("--project", type=str, default="training_results",
                       help="Project name")
    parser.add_argument("--name", type=str, default="exp",
                       help="Experiment name")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--create-configs", action="store_true",
                       help="Create default configuration files")
    
    args = parser.parse_args()
    
    # Create default configs if requested
    if args.create_configs:
        create_default_configs()
        print("✅ Default configuration files created!")
        return
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
        if not validate_config(config):
            return
    else:
        # Use command line arguments
        config = {
            'model': args.model or 'yolo11n.pt',
            'data': args.data or 'configs/datasets/my_dataset.json',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'img_size': args.img_size,
            'device': args.device,
            'project': args.project,
            'name': args.name
        }
    
    # Install YOLO
    yolo_version = get_yolo_version(config['model'])
    if not install_yolo(yolo_version):
        return
    
    # Run training
    success = run_training(config, verbose=args.verbose)
    
    if success:
        print(f"\n📋 Next steps:")
        print(f"1. Check results: {config['project']}/{config['name']}/")
        print(f"2. Monitor training: python scripts/monitor.py --project {config['project']}/{config['name']}")
        print(f"3. Run inference: python scripts/inference.py --model {config['project']}/{config['name']}/weights/best.pt")
    else:
        print("\n❌ Training failed. Check logs above for details.")

if __name__ == "__main__":
    main() 