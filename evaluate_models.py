#!/usr/bin/env python3
"""
Model Evaluation Script for Cell Detection
Evaluates and compares all trained YOLO models.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model_path, data_path, model_name):
    """Evaluate a single model."""
    print(f"Evaluating {model_name}...")
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(data=data_path)
        
        # Extract metrics
        metrics = {
            'model': model_name,
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1': results.box.map50 * 2 / (results.box.map50 + 1),  # Approximate F1
        }
        
        print(f"✅ {model_name} - mAP50: {metrics['mAP50']:.3f}, mAP50-95: {metrics['mAP50-95']:.3f}")
        return metrics
        
    except Exception as e:
        print(f"❌ Evaluation failed for {model_name}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained YOLO models for cell detection')
    parser.add_argument('--results-dir', type=str, default='training_results',
                       help='Directory containing training results')
    parser.add_argument('--data', type=str, default='datasets/cell_detection.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                       help='Output CSV file for comparison results')
    
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
    
    print(f"🔍 Evaluating models in: {args.results_dir}")
    
    # Find trained models
    results = []
    for model_name, config_path in model_configs.items():
        # Look for the best model in the results directory
        model_dir = os.path.join(args.results_dir, f"cell_detection_{model_name}")
        best_model_path = os.path.join(model_dir, "weights", "best.pt")
        
        if os.path.exists(best_model_path):
            metrics = evaluate_model(best_model_path, args.data, model_name)
            if metrics:
                results.append(metrics)
        else:
            print(f"⚠️  No trained model found for {model_name}")
    
    if not results:
        print("❌ No trained models found for evaluation")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    # Sort by mAP50 (descending)
    df = df.sort_values('mAP50', ascending=False)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n📊 Model Comparison Results:")
    print(df.to_string(index=False))
    print(f"\n💾 Results saved to: {args.output}")
    
    # Print best model
    best_model = df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['model']}")
    print(f"   mAP50: {best_model['mAP50']:.3f}")
    print(f"   mAP50-95: {best_model['mAP50-95']:.3f}")

if __name__ == "__main__":
    main() 