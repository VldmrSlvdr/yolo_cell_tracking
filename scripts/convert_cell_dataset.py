#!/usr/bin/env python3
"""
Cell Detection Dataset Conversion Script - LabelMe+CSV to YOLO format
"""

import os
import sys
import json
import argparse
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_dataset_csv(csv_path: str) -> pd.DataFrame:
    """Load dataset CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded CSV with {len(df)} annotations")
    print(f"📊 Columns: {list(df.columns)}")
    return df

def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bbox string from CSV to [x, y, w, h]"""
    try:
        # Remove brackets and split by comma
        bbox_str = bbox_str.strip('[]')
        values = [float(x.strip()) for x in bbox_str.split(',')]
        if len(values) == 4:
            return tuple(values)
        else:
            raise ValueError(f"Invalid bbox format: {bbox_str}")
    except Exception as e:
        print(f"⚠️  Error parsing bbox {bbox_str}: {e}")
        return (0.0, 0.0, 0.0, 0.0)

def convert_bbox_to_yolo(bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """Convert bbox [x, y, w, h] to YOLO format [x_center, y_center, w, h] (normalized)"""
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

def convert_labelme_to_yolo(labelme_path: str, img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """Convert LabelMe JSON to YOLO format annotations"""
    try:
        with open(labelme_path, 'r') as f:
            labelme_data = json.load(f)
        
        yolo_annotations = []
        
        if 'shapes' in labelme_data:
            for shape in labelme_data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    # Get class ID from label
                    class_id = int(shape['label']) - 1  # Convert to 0-based indexing
                    
                    # Get rectangle points
                    points = shape['points']
                    if len(points) == 2:
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        
                        # Convert to [x, y, w, h] format
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = abs(x2 - x1)
                        h = abs(y2 - y1)
                        
                        # Convert to YOLO format
                        x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo((x, y, w, h), img_width, img_height)
                        
                        yolo_annotations.append((class_id, x_center, y_center, w_norm, h_norm))
        
        return yolo_annotations
    except Exception as e:
        print(f"⚠️  Error processing LabelMe file {labelme_path}: {e}")
        return []

def convert_dataset_to_yolo(dataset_dir: str, output_dir: str, class_names: List[str] = None):
    """Convert cell detection dataset to YOLO format"""
    
    print(f"🔍 Converting Cell Detection Dataset...")
    print(f"📁 Source: {dataset_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Load dataset CSV
    csv_path = os.path.join(dataset_dir, "dataset.csv")
    df = load_dataset_csv(csv_path)
    
    # Create output directories
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_labels_dir, split), exist_ok=True)
    
    # Group annotations by image_id and source
    grouped_data = df.groupby(['image_id', 'source']).agg({
        'width': 'first',
        'height': 'first',
        'bbox': list
    }).reset_index()
    
    print(f"📊 Found {len(grouped_data)} unique images")
    
    # Process each image
    converted_count = 0
    class_counts = {}
    
    for _, row in grouped_data.iterrows():
        image_id = row['image_id']
        source = row['source']
        width = row['width']
        height = row['height']
        bboxes = row['bbox']
        
        # Determine split (map source to split)
        split = source if source in ['train', 'val', 'test'] else 'train'
        
        # Source image path
        src_image_path = os.path.join(dataset_dir, "images", split, f"{image_id}.jpg")
        if not os.path.exists(src_image_path):
            # Try other extensions
            for ext in ['.png', '.jpeg', '.tiff']:
                src_image_path = os.path.join(dataset_dir, "images", split, f"{image_id}{ext}")
                if os.path.exists(src_image_path):
                    break
        
        if not os.path.exists(src_image_path):
            print(f"⚠️  Image not found: {image_id}")
            continue
        
        # Destination paths
        dst_image_path = os.path.join(output_images_dir, split, f"{image_id}.jpg")
        label_filename = f"{image_id}.txt"
        label_path = os.path.join(output_labels_dir, split, label_filename)
        
        # Copy image
        shutil.copy2(src_image_path, dst_image_path)
        
        # Create YOLO labels from CSV bboxes
        yolo_lines = []
        for bbox_str in bboxes:
            bbox = parse_bbox(bbox_str)
            if bbox != (0.0, 0.0, 0.0, 0.0):
                # Convert bbox to YOLO format
                x_center, y_center, w, h = convert_bbox_to_yolo(bbox, width, height)
                
                # Use class 0 for cell (single class)
                class_id = 0
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
test: images/test    # subfolder under `path`

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

def main():
    parser = argparse.ArgumentParser(description="Cell Detection Dataset Conversion Script")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Path to dataset directory containing images/, annotations/, and dataset.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for YOLO format dataset")
    parser.add_argument("--dataset-name", type=str, default="cell_detection",
                       help="Dataset name for YAML config")
    parser.add_argument("--class-names", type=str, nargs='+', default=["cell"],
                       help="Class names (default: cell)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        return
    
    csv_path = os.path.join(args.dataset_dir, "dataset.csv")
    if not os.path.exists(csv_path):
        print(f"❌ dataset.csv not found in: {args.dataset_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert dataset
    class_counts = convert_dataset_to_yolo(args.dataset_dir, args.output_dir, args.class_names)
    
    # Create YAML config
    yaml_path = create_yaml_config(args.output_dir, args.dataset_name, args.class_names)
    
    print(f"✅ Dataset conversion complete!")
    print(f"📁 Output structure:")
    print(f"   {args.output_dir}/")
    print(f"   ├── images/")
    print(f"   │   ├── train/")
    print(f"   │   ├── val/")
    print(f"   │   └── test/")
    print(f"   ├── labels/")
    print(f"   │   ├── train/")
    print(f"   │   ├── val/")
    print(f"   │   └── test/")
    print(f"   └── {args.dataset_name}.yaml")

if __name__ == "__main__":
    main()



