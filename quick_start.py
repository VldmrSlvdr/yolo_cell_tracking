#!/usr/bin/env python3
"""
Quick Start Script for YOLO Training Repository
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"🔧 Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Command completed successfully!")
        if result.stdout:
            print("📊 Output:")
            print(result.stdout)
    else:
        print("❌ Command failed!")
        print("📋 Error:")
        print(result.stderr)
        return False
    
    return True

def setup_environment():
    """Setup the training environment"""
    print("🔧 Setting up YOLO training environment...")
    
    # Install requirements
    cmd = f"{sys.executable} -m pip install ultralytics torch torchvision"
    if not run_command(cmd, "Installing YOLO and PyTorch"):
        return False
    
    # Create default configs
    cmd = f"{sys.executable} scripts/train.py --create-configs"
    if not run_command(cmd, "Creating default configuration files"):
        return False
    
    print("✅ Environment setup complete!")
    return True

def convert_cell_dataset():
    """Convert cell detection dataset"""
    print("📚 Converting cell detection dataset...")
    
    cmd = f"{sys.executable} scripts/convert_dataset.py --config configs/datasets/cell_detection.json"
    if not run_command(cmd, "Converting COCO dataset to YOLO format"):
        return False
    
    print("✅ Dataset conversion complete!")
    return True

def train_cell_detection():
    """Train cell detection model"""
    print("🚀 Training cell detection model...")
    
    cmd = f"{sys.executable} scripts/train.py --config configs/training/cell_detection.json"
    if not run_command(cmd, "Training YOLO11n for cell detection"):
        return False
    
    print("✅ Training complete!")
    return True

def test_inference():
    """Test inference on trained model"""
    print("🔍 Testing inference...")
    
    # Find the best model
    model_path = "training_results/cell_detection_yolo11n/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Test inference
    cmd = f"{sys.executable} scripts/inference.py --model {model_path} --search C001T054"
    if not run_command(cmd, "Testing inference on cell image"):
        return False
    
    print("✅ Inference test complete!")
    return True

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Convert dataset", "python scripts/convert_dataset.py --config configs/datasets/cell_detection.json"),
        ("Train model", "python scripts/train.py --config configs/training/cell_detection.json"),
        ("Monitor training", "python scripts/monitor.py --project training_results/cell_detection_yolo11n --monitor"),
        ("Run inference", "python scripts/inference.py --model training_results/cell_detection_yolo11n/weights/best.pt --search image_name"),
        ("Batch inference", "python scripts/inference.py --model training_results/cell_detection_yolo11n/weights/best.pt --source datasets/cell_detection/images/val/"),
        ("Validate model", "python scripts/inference.py --model training_results/cell_detection_yolo11n/weights/best.pt --validate --data datasets/cell_detection/cell_detection.yaml"),
        ("Generate plots", "python scripts/monitor.py --project training_results/cell_detection_yolo11n --plots"),
        ("Create report", "python scripts/monitor.py --project training_results/cell_detection_yolo11n --report")
    ]
    
    for description, command in examples:
        print(f"📋 {description}:")
        print(f"   {command}")
        print()

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Repository Quick Start")
    parser.add_argument("--setup", action="store_true",
                       help="Setup environment and create configs")
    parser.add_argument("--convert", action="store_true",
                       help="Convert cell detection dataset")
    parser.add_argument("--train", action="store_true",
                       help="Train cell detection model")
    parser.add_argument("--test", action="store_true",
                       help="Test inference")
    parser.add_argument("--full", action="store_true",
                       help="Run complete workflow (setup + convert + train + test)")
    parser.add_argument("--examples", action="store_true",
                       help="Show usage examples")
    
    args = parser.parse_args()
    
    print("🚀 YOLO Training Repository Quick Start")
    print("=" * 50)
    
    if args.examples:
        show_usage_examples()
        return
    
    if args.setup:
        setup_environment()
        return
    
    if args.convert:
        convert_cell_dataset()
        return
    
    if args.train:
        train_cell_detection()
        return
    
    if args.test:
        test_inference()
        return
    
    if args.full:
        print("🔄 Running complete workflow...")
        
        if not setup_environment():
            print("❌ Setup failed!")
            return
        
        if not convert_cell_dataset():
            print("❌ Dataset conversion failed!")
            return
        
        if not train_cell_detection():
            print("❌ Training failed!")
            return
        
        if not test_inference():
            print("❌ Inference test failed!")
            return
        
        print("\n🎉 Complete workflow finished successfully!")
        print("📋 Next steps:")
        print("1. Check results: training_results/cell_detection_yolo11n/")
        print("2. Run inference: python scripts/inference.py --model training_results/cell_detection_yolo11n/weights/best.pt --search image_name")
        print("3. Monitor training: python scripts/monitor.py --project training_results/cell_detection_yolo11n --monitor")
        return
    
    # Default: show usage
    print("💡 Usage:")
    print("   python quick_start.py --setup     # Setup environment")
    print("   python quick_start.py --convert   # Convert dataset")
    print("   python quick_start.py --train     # Train model")
    print("   python quick_start.py --test      # Test inference")
    print("   python quick_start.py --full      # Complete workflow")
    print("   python quick_start.py --examples  # Show examples")

if __name__ == "__main__":
    main() 