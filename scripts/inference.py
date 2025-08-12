#!/usr/bin/env python3
"""
YOLO11 inference for cell detection
"""

import os
import sys
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

def find_image_in_dataset(image_name: str, search_paths: list[str] = None):
    """Find image in dataset directories"""
    
    if search_paths is None:
        search_paths = [
            "datasets/*/images/train",
            "datasets/*/images/val",
            "datasets/*/images/*",
            ".",
            ".."
        ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and image_name.lower() in file.lower():
                        return os.path.join(root, file)
    return None

def run_inference(model_path: str, 
                 image_path: str, 
                 confidence: float = 0.25,
                 device: str = "0",
                 save_dir: str = None,
                 show_labels: bool = True):
    """Run YOLO11 inference on an image using Python API"""
    
    print(f"🔍 Running YOLO11 inference...")
    print(f"🤖 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    print(f"🎯 Confidence: {confidence}")
    print(f"💻 Device: {device}")
    print(f"🏷️  Show labels: {show_labels}")
    
    # Determine output directory (don't create it yet, let YOLO handle it)
    if save_dir:
        project_dir = save_dir
        run_name = "inference"
    else:
        project_dir = "runs/detect"
        run_name = "predict"
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=confidence,
        device=device,
        save=False,  # Don't auto-save images, we'll handle it
        save_txt=True,  # Save label txt files
        save_conf=True,  # Save confidence in txt files
        project=project_dir,  # Set project dir
        name=run_name  # Set run name
    )
    
    # Get the actual save directory from results
    actual_save_dir = None
    if results and hasattr(results[0], 'save_dir'):
        actual_save_dir = str(results[0].save_dir)
    
    # Save visualization with/without labels
    for result in results:
        # Use the same directory as YOLO used for labels
        if actual_save_dir:
            output_path = os.path.join(actual_save_dir, os.path.basename(image_path))
        else:
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, filename)
        
        # Plot with configured options
        im = result.plot(labels=show_labels, conf=show_labels, boxes=True)
        
        # Save image
        cv2.imwrite(output_path, im)
        
        print("✅ Inference completed successfully!")
        if actual_save_dir:
            print(f"📁 Results saved to: {actual_save_dir}")
            print(f"   📄 {os.path.basename(image_path)}")
            
            # Check for label files in the same directory
            labels_dir = os.path.join(actual_save_dir, "labels")
            if os.path.exists(labels_dir):
                print(f"📁 Labels saved to: {labels_dir}")
                for label_file in os.listdir(labels_dir):
                    if label_file.endswith('.txt'):
                        print(f"   📄 {label_file}")
        else:
            print(f"📁 Results saved to: {output_dir}")
            print(f"   📄 {os.path.basename(image_path)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="YOLO11 inference for cell detection")
    parser.add_argument("--model", type=str, 
                       default="cell_detection_yolo11/exp/weights/best.pt",
                       help="Path to trained YOLO11 model")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to input image")
    parser.add_argument("--search", type=str, default=None,
                       help="Search for image by name in dataset directories")
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold (default: 0.25)")
    parser.add_argument("--device", type=str, default="0",
                       help="Device to use (0 for GPU, cpu for CPU)")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save results")
    parser.add_argument("--label_switch", action="store_true",
                       help="Turn off cell labels in output images (show only bounding boxes)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("💡 Please train a model first:")
        print("   python train_yolo11.py")
        return
    
    # Determine image path
    image_path = None
    if args.search:
        print(f"🔍 Searching for image: {args.search}")
        image_path = find_image_in_dataset(args.search)
        if image_path:
            print(f"✅ Found image: {image_path}")
        else:
            print(f"❌ Image not found: {args.search}")
            print("💡 Searched in:")
            print("   - datasets/*/images/train")
            print("   - datasets/*/images/val")
            print("   - Current directory")
            return
    elif args.image:
        image_path = args.image
    else:
        print("❌ Please specify either --image or --search")
        print("💡 Usage examples:")
        print("   python inference.py --image path/to/image.jpg")
        print("   python inference.py --search Aequalized_Image_01_01_01_01_C001T054.jpg")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Run inference
    success = run_inference(
        model_path=args.model,
        image_path=image_path,
        confidence=args.confidence,
        device=args.device,
        save_dir=args.save_dir,
        show_labels=not args.label_switch  # Invert the switch
    )

if __name__ == "__main__":
    main()