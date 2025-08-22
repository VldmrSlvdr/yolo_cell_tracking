#!/usr/bin/env python3
"""
Comprehensive Model Comparison and Visualization Script
Compares performance metrics between different YOLO models and creates GT vs Prediction visualizations.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

def set_plot_style():
    """Set plotting style for consistent visualization."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

def load_results_csv(exp_dir):
    """Load results.csv from experiment directory."""
    results_path = os.path.join(exp_dir, 'results.csv')
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        # Get the best epoch metrics
        best_idx = df['metrics/mAP50(B)'].idxmax()
        return df.iloc[best_idx]
    return None

def extract_model_metrics(exp_dirs, model_names):
    """Extract metrics from all experiment directories."""
    results = []
    
    for exp_dir, model_name in zip(exp_dirs, model_names):
        if not os.path.exists(exp_dir):
            print(f"⚠️  Experiment directory not found: {exp_dir}")
            continue
            
        # Load metrics from results.csv
        metrics = load_results_csv(exp_dir)
        
        if metrics is not None:
            result = {
                'model': model_name,
                'mAP50': metrics['metrics/mAP50(B)'],
                'mAP50-95': metrics['metrics/mAP50-95(B)'],
                'precision': metrics['metrics/precision(B)'],
                'recall': metrics['metrics/recall(B)'],
                'train_loss': metrics['train/box_loss'] + metrics['train/cls_loss'] + metrics['train/dfl_loss'],
                'val_loss': metrics['val/box_loss'] + metrics['val/cls_loss'] + metrics['val/dfl_loss'],
                'best_epoch': int(metrics['epoch']),
                'exp_dir': exp_dir
            }
            results.append(result)
            print(f"✅ Loaded metrics for {model_name}")
        else:
            print(f"❌ Could not load metrics for {model_name}")
    
    return results

def create_performance_comparison(results, output_dir):
    """Create comprehensive performance comparison visualizations."""
    if not results:
        print("❌ No results to visualize")
        return
    
    df = pd.DataFrame(results)
    
    # Set plot style
    set_plot_style()
    
    # 1. Overall Performance Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=20, fontweight='bold')
    
    # mAP50 comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['model'], df['mAP50'], color='skyblue', edgecolor='navy', linewidth=1.5)
    ax1.set_title('mAP@0.5', fontweight='bold')
    ax1.set_ylabel('mAP@0.5')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # mAP50-95 comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['model'], df['mAP50-95'], color='lightcoral', edgecolor='darkred', linewidth=1.5)
    ax2.set_title('mAP@0.5:0.95', fontweight='bold')
    ax2.set_ylabel('mAP@0.5:0.95')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision vs Recall
    ax3 = axes[1, 0]
    ax3.scatter(df['recall'], df['precision'], c=df['mAP50'], s=200, alpha=0.7, 
               cmap='viridis', edgecolors='black', linewidth=1.5)
    for i, model in enumerate(df['model']):
        ax3.annotate(model, (df['recall'].iloc[i], df['precision'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision vs Recall (colored by mAP@0.5)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Loss comparison
    ax4 = axes[1, 1]
    x = np.arange(len(df['model']))
    width = 0.35
    bars_train = ax4.bar(x - width/2, df['train_loss'], width, label='Train Loss', 
                        color='orange', alpha=0.7, edgecolor='darkorange', linewidth=1.5)
    bars_val = ax4.bar(x + width/2, df['val_loss'], width, label='Val Loss', 
                      color='green', alpha=0.7, edgecolor='darkgreen', linewidth=1.5)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training vs Validation Loss', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['model'], rotation=45)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Detailed metrics table
    create_metrics_table(df, output_dir)
    
    # 3. Ranking visualization
    create_ranking_visualization(df, output_dir)
    
    print(f"✅ Performance comparison charts saved to {output_dir}")

def create_metrics_table(df, output_dir):
    """Create a detailed metrics comparison table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = df[['model', 'mAP50', 'mAP50-95', 'precision', 'recall', 'best_epoch']].copy()
    table_data = table_data.round(4)
    table_data.columns = ['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'Best Epoch']
    
    # Sort by mAP50
    table_data = table_data.sort_values('mAP@0.5', ascending=False)
    
    # Create table
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color the header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the best performing model
    for i in range(len(table_data.columns)):
        table[(1, i)].set_facecolor('#E8F5E8')
    
    plt.title('Detailed Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'metrics_table.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_ranking_visualization(df, output_dir):
    """Create model ranking visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by mAP50
    df_sorted = df.sort_values('mAP50', ascending=True)
    
    # Create horizontal bar chart
    bars = ax.barh(df_sorted['model'], df_sorted['mAP50'], color='lightblue', 
                   edgecolor='navy', linewidth=1.5)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df_sorted['mAP50'])):
        ax.text(value + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
    
    ax.set_xlabel('mAP@0.5')
    ax.set_title('Model Ranking by mAP@0.5', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add rank numbers
    for i, model in enumerate(df_sorted['model']):
        rank = len(df_sorted) - i
        ax.text(-0.02, i, f'#{rank}', va='center', ha='right', 
                fontweight='bold', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_ranking.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_gt_vs_pred_visualization(exp_dirs, model_names, output_dir):
    """Create GT vs Prediction visualization for each model."""
    print("\n🎨 Creating GT vs Prediction visualizations...")
    
    for exp_dir, model_name in zip(exp_dirs, model_names):
        if not os.path.exists(exp_dir):
            continue
            
        # Look for validation batch images
        gt_images = []
        pred_images = []
        
        for i in range(3):  # Check first 3 validation batches
            gt_path = os.path.join(exp_dir, f'val_batch{i}_labels.jpg')
            pred_path = os.path.join(exp_dir, f'val_batch{i}_pred.jpg')
            
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                gt_images.append(gt_path)
                pred_images.append(pred_path)
        
        if not gt_images:
            print(f"⚠️  No validation images found for {model_name}")
            continue
        
        # Create comparison figure
        fig, axes = plt.subplots(len(gt_images), 2, figsize=(16, 6*len(gt_images)))
        if len(gt_images) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{model_name} - Ground Truth vs Predictions', fontsize=16, fontweight='bold')
        
        for i, (gt_path, pred_path) in enumerate(zip(gt_images, pred_images)):
            # Load images
            gt_img = Image.open(gt_path)
            pred_img = Image.open(pred_path)
            
            # Display GT
            axes[i, 0].imshow(gt_img)
            axes[i, 0].set_title(f'Ground Truth - Batch {i}', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Display Predictions
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f'Predictions - Batch {i}', fontweight='bold')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_gt_vs_pred.png'), 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ GT vs Pred visualization created for {model_name}")

def save_comparison_csv(results, output_dir):
    """Save detailed comparison results to CSV."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('mAP50', ascending=False)
    
    # Add rank column
    df['rank'] = range(1, len(df) + 1)
    
    # Reorder columns
    columns = ['rank', 'model', 'mAP50', 'mAP50-95', 'precision', 'recall', 'train_loss', 'val_loss', 'best_epoch']
    df = df[columns]
    
    output_path = os.path.join(output_dir, 'model_comparison_detailed.csv')
    df.to_csv(output_path, index=False)
    print(f"✅ Detailed comparison saved to {output_path}")
    
    return df

def print_summary(results):
    """Print a summary of the comparison results."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    df = df.sort_values('mAP50', ascending=False)
    
    print("\n" + "="*60)
    print("🏆 MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"📊 Total models evaluated: {len(df)}")
    print(f"🥇 Best model: {df.iloc[0]['model']}")
    print(f"   mAP@0.5: {df.iloc[0]['mAP50']:.4f}")
    print(f"   mAP@0.5:0.95: {df.iloc[0]['mAP50-95']:.4f}")
    print(f"   Precision: {df.iloc[0]['precision']:.4f}")
    print(f"   Recall: {df.iloc[0]['recall']:.4f}")
    
    print(f"\n📈 Performance range:")
    print(f"   mAP@0.5: {df['mAP50'].min():.4f} - {df['mAP50'].max():.4f}")
    print(f"   mAP@0.5:0.95: {df['mAP50-95'].min():.4f} - {df['mAP50-95'].max():.4f}")
    
    print(f"\n📋 Top 3 models:")
    for i in range(min(3, len(df))):
        model = df.iloc[i]
        print(f"   {i+1}. {model['model']}: mAP@0.5={model['mAP50']:.4f}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Compare YOLO model performance and create visualizations')
    parser.add_argument('--exp-dir', type=str, default='/mnt/d/exp_results/cell_track/yolo',
                       help='Base directory containing experiment results')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['yolov8n', 'yolov8s', 'yolov8m', 'yolo11n', 'yolo11s', 'yolo11m'],
                       help='Models to compare')
    parser.add_argument('--output-dir', type=str, default='model_comparison_results',
                       help='Output directory for comparison results')
    parser.add_argument('--create-visualizations', action='store_true', default=True,
                       help='Create GT vs Prediction visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Starting model comparison...")
    print(f"📁 Experiment directory: {args.exp_dir}")
    print(f"🎯 Models to compare: {', '.join(args.models)}")
    print(f"📤 Output directory: {args.output_dir}")
    
    # Map model names to experiment directories
    exp_dirs = []
    model_names = []
    
    for model in args.models:
        # Try different naming patterns
        possible_names = [
            f"cell_detection_{model}",
            f"cell_detection_{model}_16bs",
            f"{model}",
        ]
        
        found = False
        for name in possible_names:
            exp_path = os.path.join(args.exp_dir, name)
            if os.path.exists(exp_path):
                exp_dirs.append(exp_path)
                model_names.append(model)
                found = True
                break
        
        if not found:
            print(f"⚠️  Experiment directory not found for {model}")
    
    if not exp_dirs:
        print("❌ No experiment directories found")
        return
    
    # Extract metrics
    print(f"\n📊 Extracting metrics from {len(exp_dirs)} experiments...")
    results = extract_model_metrics(exp_dirs, model_names)
    
    if not results:
        print("❌ No valid results found")
        return
    
    # Create performance comparison
    print(f"\n📈 Creating performance comparison charts...")
    create_performance_comparison(results, args.output_dir)
    
    # Create GT vs Prediction visualizations
    if args.create_visualizations:
        create_gt_vs_pred_visualization(exp_dirs, model_names, args.output_dir)
    
    # Save detailed results
    df = save_comparison_csv(results, args.output_dir)
    
    # Print summary
    print_summary(results)
    
    print(f"\n🎉 Model comparison completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

