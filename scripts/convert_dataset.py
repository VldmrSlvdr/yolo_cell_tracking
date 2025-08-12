#!/usr/bin/env python3
"""
Dataset Conversion Script - COCO to YOLO format
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """Load dataset configuration"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def convert_bbox_to_yolo(bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert COCO bbox to YOLO format"""
    x, y, w, h = bbox
    
    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def convert_coco_to_yolo(coco_annotations_path: str, 
                         images_dir: str, 
                         output_dir: str,
                         class_mapping: Dict[int, str] = None,
                         class_id_start: int = 0):
    """Convert COCO format to YOLO format"""
    
    print(f"🔍 Converting COCO annotations...")
    print(f"📁 Source: {coco_annotations_path}")
    print(f"📁 Images: {images_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_labels_dir, split), exist_ok=True)
    
    # Load COCO annotations
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to image mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    print(f"📊 Found {len(coco_data['images'])} images")
    print(f"📊 Found {len(coco_data['annotations'])} annotations")
    
    # Process each image
    converted_count = 0
    class_counts = {}
    
    for image_info in coco_data['images']:
        image_id = image_info['id']
        filename = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        
        # Determine split
        split = "train" if "train" in coco_annotations_path else "val"
        
        # Source image path
        src_image_path = os.path.join(images_dir, split, filename)
        
        if not os.path.exists(src_image_path):
            print(f"⚠️  Image not found: {src_image_path}")
            continue
        
        # Destination paths
        dst_image_path = os.path.join(output_images_dir, split, filename)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(output_labels_dir, split, label_filename)
        
        # Copy image
        shutil.copy2(src_image_path, dst_image_path)
        
        # Create YOLO labels
        yolo_lines = []
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                bbox = ann['bbox']
                category_id = ann['category_id']
                
                # Map class ID
                if class_mapping:
                    if category_id in class_mapping:
                        class_id = class_mapping[category_id] - class_id_start
                    else:
                        continue
                else:
                    class_id = category_id - class_id_start
                
                # Convert bbox
                x_center, y_center, w, h = convert_bbox_to_yolo(bbox, width, height)
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                yolo_lines.append(yolo_line)
                
                # Count classes
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        # Write label file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        converted_count += 1
        if converted_count % 100 == 0:
            print(f"✅ Converted {converted_count} images...")
    
    print(f"✅ Conversion complete! Converted {converted_count} images")
    print(f"📊 Class distribution: {class_counts}")
    
    return class_counts

def create_yaml_config(output_dir: str, dataset_name: str, class_names: List[str]):
    """Create YAML configuration file"""
    yaml_content = f"""# {dataset_name}.yaml

path: {output_dir}  # root dir of your dataset
train: images/train  # subfolder under `path`
val: images/val      # subfolder under `path`

# number of classes
nc: {len(class_names)}

# class names
names: {class_names}
"""
    
    yaml_path = os.path.join(output_dir, f"{dataset_name}.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"📄 Created YAML config: {yaml_path}")
    return yaml_path

def convert_dataset_with_config(config_path: str):
    """Convert dataset using configuration file"""
    
    config = load_dataset_config(config_path)
    
    print(f"🔍 Converting dataset: {config['name']}")
    print(f"📁 Source annotations: {config['coco_annotations']}")
    print(f"📁 Source images: {config['images_dir']}")
    print(f"📁 Output: {config['output_dir']}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Convert training set
    if 'train_annotations' in config:
        print("\n📚 Converting training set...")
        train_class_counts = convert_coco_to_yolo(
            config['train_annotations'],
            config['images_dir'],
            config['output_dir'],
            config.get('class_mapping'),
            config.get('class_id_start', 0)
        )
    
    # Convert validation set
    if 'val_annotations' in config:
        print("\n📚 Converting validation set...")
        val_class_counts = convert_coco_to_yolo(
            config['val_annotations'],
            config['images_dir'],
            config['output_dir'],
            config.get('class_mapping'),
            config.get('class_id_start', 0)
        )
    
    # Create YAML config
    class_names = config.get('class_names', ['object'])
    yaml_path = create_yaml_config(config['output_dir'], config['name'], class_names)
    
    # Create dataset config for training
    dataset_config = {
        'name': config['name'],
        'path': config['output_dir'],
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names,
        'description': config.get('description', 'Converted dataset')
    }
    
    dataset_config_path = os.path.join(config['output_dir'], f"{config['name']}_config.json")
    with open(dataset_config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"✅ Dataset conversion complete!")
    print(f"📁 Output structure:")
    print(f"   {config['output_dir']}/")
    print(f"   ├── images/")
    print(f"   │   ├── train/")
    print(f"   │   └── val/")
    print(f"   ├── labels/")
    print(f"   │   ├── train/")
    print(f"   │   └── val/")
    print(f"   ├── {config['name']}.yaml")
    print(f"   └── {config['name']}_config.json")

def interactive_converter():
    """Interactive dataset converter"""
    
    print("🔍 Interactive Dataset Converter")
    print("=" * 40)
    
    # Get dataset information
    dataset_name = input("📝 Dataset name: ").strip()
    coco_annotations = input("📁 COCO annotations path: ").strip()
    images_dir = input("📁 Images directory path: ").strip()
    output_dir = input("📁 Output directory: ").strip()
    
    # Get class information
    num_classes = int(input("🔢 Number of classes: "))
    class_names = []
    for i in range(num_classes):
        class_name = input(f"🏷️  Class {i} name: ").strip()
        class_names.append(class_name)
    
    # Create config
    config = {
        'name': dataset_name,
        'coco_annotations': coco_annotations,
        'images_dir': images_dir,
        'output_dir': output_dir,
        'class_names': class_names,
        'description': f'Converted {dataset_name} dataset'
    }
    
    # Save config
    config_path = f"configs/datasets/{dataset_name}.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📄 Config saved to: {config_path}")
    
    # Convert dataset
    convert_dataset_with_config(config_path)

def main():
    parser = argparse.ArgumentParser(description="Dataset Conversion Script")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to dataset configuration file")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode")
    parser.add_argument("--coco-annotations", type=str, default=None,
                       help="Path to COCO annotations file")
    parser.add_argument("--images-dir", type=str, default=None,
                       help="Path to images directory")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--class-names", type=str, nargs='+', default=None,
                       help="Class names")
    parser.add_argument("--dataset-name", type=str, default=None,
                       help="Dataset name")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_converter()
    elif args.config:
        convert_dataset_with_config(args.config)
    elif args.coco_annotations and args.images_dir and args.output_dir:
        # Quick conversion
        if not args.class_names:
            args.class_names = ['object']
        if not args.dataset_name:
            args.dataset_name = 'my_dataset'
        
        config = {
            'name': args.dataset_name,
            'coco_annotations': args.coco_annotations,
            'images_dir': args.images_dir,
            'output_dir': args.output_dir,
            'class_names': args.class_names
        }
        
        # Save temporary config
        temp_config = f"temp_{args.dataset_name}.json"
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        convert_dataset_with_config(temp_config)
        
        # Clean up
        os.remove(temp_config)
    else:
        print("❌ Please provide either --config, --interactive, or all required arguments")
        print("💡 Usage examples:")
        print("   python convert_dataset.py --config configs/datasets/my_dataset.json")
        print("   python convert_dataset.py --interactive")
        print("   python convert_dataset.py --coco-annotations path/to/annotations.json --images-dir path/to/images --output-dir datasets/my_dataset --class-names cell")

if __name__ == "__main__":
    main() 