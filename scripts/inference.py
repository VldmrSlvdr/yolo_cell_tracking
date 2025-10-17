#!/usr/bin/env python3
"""
YOLO11 inference for cell detection
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def find_image_in_dataset(image_name: str, search_paths: list[str] = None):
    """Find image in dataset directories"""
    
    if search_paths is None:
        search_paths = [
            "datasets/*/images/train",
            "datasets/*/images/val",
            "datasets/*/images/*",
            "test_image",
            ".",
            ".."
        ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                # Skip runs directories (they contain processed outputs, not original images)
                if 'runs' in root or 'results' in root:
                    continue
                    
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and image_name.lower() in file.lower():
                        return os.path.join(root, file)
    return None

def find_label_file(image_path: str):
    """Find corresponding label file for an image"""
    # Try to find label file in typical YOLO dataset structure
    image_path = Path(image_path)
    
    print(f"🔍 Searching for GT labels for: {image_path}")
    
    # Replace 'images' with 'labels' in path and change extension to .txt
    label_path = str(image_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
    label_path = Path(label_path).with_suffix('.txt')
    
    print(f"   Trying: {label_path}")
    if label_path.exists():
        return str(label_path)
    
    # Try sibling directory
    parent = image_path.parent.parent
    label_dir = parent / 'labels' / image_path.parent.name
    label_path = label_dir / image_path.with_suffix('.txt').name
    
    print(f"   Trying: {label_path}")
    if label_path.exists():
        return str(label_path)
    
    print(f"   ❌ Label file not found in standard locations")
    return None

def load_yolo_labels(label_path: str, img_width: int, img_height: int):
    """Load YOLO format labels and convert to pixel coordinates
    
    Returns:
        list of dicts with keys: class_id, x1, y1, x2, y2 (in pixels)
    """
    boxes = []
    
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert from normalized to pixel coordinates
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                boxes.append({
                    'class_id': class_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
    
    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes
    
    Args:
        box1, box2: dicts with keys x1, y1, x2, y2
    
    Returns:
        float: IoU value
    """
    # Calculate intersection
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes
    
    Returns:
        tuple: (matched_pred_indices, matched_gt_indices, unmatched_pred_indices)
    """
    matched_pred = []
    matched_gt = []
    unmatched_pred = []
    
    used_gt = set()
    
    # Sort predictions by confidence (if available)
    pred_indices = list(range(len(pred_boxes)))
    if pred_boxes and 'conf' in pred_boxes[0]:
        pred_indices.sort(key=lambda i: pred_boxes[i]['conf'], reverse=True)
    
    for pred_idx in pred_indices:
        pred_box = pred_boxes[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            
            # Only match boxes of the same class
            if pred_box['class_id'] != gt_box['class_id']:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_pred.append(pred_idx)
            matched_gt.append(best_gt_idx)
            used_gt.add(best_gt_idx)
        else:
            unmatched_pred.append(pred_idx)
    
    return matched_pred, matched_gt, unmatched_pred

def draw_comparison_boxes(image, pred_boxes, gt_boxes, iou_threshold=0.5, show_labels=True):
    """Draw boxes with color coding: blue=GT (FN), green=TP, red=FP
    
    Args:
        image: numpy array (BGR)
        pred_boxes: list of predicted boxes with keys: class_id, x1, y1, x2, y2, conf, class_name
        gt_boxes: list of GT boxes with keys: class_id, x1, y1, x2, y2
        iou_threshold: IoU threshold for matching (default: 0.5)
        show_labels: whether to show labels
    
    Returns:
        numpy array: image with drawn boxes
    """
    img = image.copy()
    
    # Match predictions to GT
    matched_pred, matched_gt, unmatched_pred = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold)
    
    # Calculate unmatched GT boxes
    unmatched_gt = set(range(len(gt_boxes))) - set(matched_gt)
    
    # Print statistics
    print(f"   📊 Matching results:")
    print(f"      • True Positives (green): {len(matched_pred)}")
    print(f"      • False Positives (red): {len(unmatched_pred)}")
    print(f"      • False Negatives (blue): {len(unmatched_gt)}")
    
    # Draw GT boxes (blue) - draw unmatched GT boxes (False Negatives)
    for gt_idx in unmatched_gt:
        box = gt_boxes[gt_idx]
        cv2.rectangle(img, (box['x1'], box['y1']), (box['x2'], box['y2']), 
                     (255, 0, 0), 2)  # Blue
        if show_labels:
            label = f"FN"
            cv2.putText(img, label, (box['x1'], box['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw true positive boxes (green)
    for pred_idx in matched_pred:
        box = pred_boxes[pred_idx]
        cv2.rectangle(img, (box['x1'], box['y1']), (box['x2'], box['y2']), 
                     (0, 255, 0), 2)  # Green
        if show_labels:
            conf = box.get('conf', 0.0)
            class_name = box.get('class_name', str(box['class_id']))
            label = f"TP {conf:.2f}"
            cv2.putText(img, label, (box['x1'], box['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw false positive boxes (red)
    for pred_idx in unmatched_pred:
        box = pred_boxes[pred_idx]
        cv2.rectangle(img, (box['x1'], box['y1']), (box['x2'], box['y2']), 
                     (0, 0, 255), 2)  # Red
        if show_labels:
            conf = box.get('conf', 0.0)
            class_name = box.get('class_name', str(box['class_id']))
            label = f"FP {conf:.2f}"
            cv2.putText(img, label, (box['x1'], box['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add legend with counts
    legend_y = 30
    cv2.putText(img, f"Blue: False Negatives ({len(unmatched_gt)})", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, f"Green: True Positives ({len(matched_pred)})", (10, legend_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Red: False Positives ({len(unmatched_pred)})", (10, legend_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return img

def run_inference(model_path: str, 
                 image_path: str, 
                 confidence: float = 0.25,
                 device: str = "0",
                 save_dir: str = None,
                 show_labels: bool = True,
                 compare_gt: bool = False,
                 iou_threshold: float = 0.5,
                 max_det: int = None):
    """Run YOLO11 inference on an image using Python API"""
    
    # Try to read max_det from model's args.yaml if not specified
    if max_det is None:
        try:
            # Model path is typically: training_results/.../weights/best.pt
            # Args.yaml is at: training_results/.../args.yaml
            model_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
            args_yaml_path = os.path.join(model_dir, 'args.yaml')
            
            if os.path.exists(args_yaml_path):
                import yaml
                with open(args_yaml_path, 'r') as f:
                    args_yaml = yaml.safe_load(f)
                    if 'max_det' in args_yaml:
                        max_det = args_yaml['max_det']
                        print(f"📋 Loaded max_det={max_det} from model's args.yaml")
        except Exception as e:
            pass
    
    # Fallback to default if still None
    if max_det is None:
        max_det = 300
        print(f"⚠️  Using default max_det=300")
    
    print(f"🔍 Running YOLO11 inference...")
    print(f"🤖 Model: {model_path}")
    print(f"🖼️  Image: {image_path}")
    print(f"🎯 Confidence: {confidence}")
    print(f"🎯 Max detections: {max_det}")
    print(f"💻 Device: {device}")
    print(f"🏷️  Show labels: {show_labels}")
    if compare_gt:
        print(f"📊 GT comparison mode: enabled (IoU threshold: {iou_threshold})")
    
    # Determine output directory (don't create it yet, let YOLO handle it)
    if save_dir:
        project_dir = save_dir
        run_name = "inference"
    else:
        project_dir = "runs/detect"
        run_name = "predict"
    
    # Load model
    model = YOLO(model_path)
    
    # Load GT boxes if comparison mode is enabled (BEFORE inference changes paths)
    gt_boxes = []
    original_image_path = image_path  # Store original path
    if compare_gt:
        label_path = find_label_file(original_image_path)
        if label_path:
            print(f"📄 Found GT labels: {label_path}")
            # Load original image to get dimensions
            img = cv2.imread(original_image_path)
            img_height, img_width = img.shape[:2]
            gt_boxes = load_yolo_labels(label_path, img_width, img_height)
            print(f"📦 Loaded {len(gt_boxes)} GT boxes")
        else:
            print("⚠️  No GT labels found for comparison")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=confidence,
        device=device,
        max_det=max_det,  # Maximum number of detections per image
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
        
        # Generate visualization
        if compare_gt and len(gt_boxes) > 0:
            # Use comparison visualization
            img = cv2.imread(original_image_path)
            
            # Extract predicted boxes from result
            pred_boxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_data = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes_data)):
                    box = boxes_data[i]
                    pred_boxes.append({
                        'class_id': int(classes[i]),
                        'x1': int(box[0]),
                        'y1': int(box[1]),
                        'x2': int(box[2]),
                        'y2': int(box[3]),
                        'conf': float(confs[i]),
                        'class_name': model.names[classes[i]]
                    })
            
            print(f"📦 Found {len(pred_boxes)} predicted boxes vs {len(gt_boxes)} GT boxes")
            print(f"🎨 Drawing comparison visualization...")
            im = draw_comparison_boxes(img, pred_boxes, gt_boxes, iou_threshold, show_labels)
            print(f"✅ Comparison visualization complete")
        else:
            # Use default YOLO visualization
            if compare_gt:
                print("⚠️  Using default visualization (no GT boxes available)")
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
    parser.add_argument("--max-det", type=int, default=None,
                       help="Maximum number of detections per image (default: auto-detect from model's args.yaml or 300)")
    parser.add_argument("--device", type=str, default="0",
                       help="Device to use (0 for GPU, cpu for CPU)")
    parser.add_argument("--save-dir", type=str, default=None,
                       help="Directory to save results")
    parser.add_argument("--label_switch", action="store_true",
                       help="Turn off cell labels in output images (show only bounding boxes)")
    parser.add_argument("--compare-gt", action="store_true",
                       help="Compare predictions with ground truth (GT=blue, TP=green, FP=red)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for matching predictions to GT (default: 0.5)")
    
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
        show_labels=not args.label_switch,  # Invert the switch
        compare_gt=args.compare_gt,
        iou_threshold=args.iou_threshold,
        max_det=args.max_det
    )

if __name__ == "__main__":
    main()