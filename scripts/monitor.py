#!/usr/bin/env python3
"""
YOLO Training Monitor - Real-time progress tracking
"""

import os
import sys
import json
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Optional

def monitor_training_logs(project_dir: str):
    """Monitor training logs and provide real-time feedback"""
    
    print(f"🔍 Monitoring YOLO training...")
    print(f"📁 Project: {project_dir}")
    print("📊 Real-time metrics will be displayed below:")
    print("=" * 60)
    
    # Check if training directory exists
    if not os.path.exists(project_dir):
        print(f"❌ Training directory not found: {project_dir}")
        print("💡 Training may not have started yet...")
        return
    
    # Monitor results.csv for metrics
    results_file = os.path.join(project_dir, "results.csv")
    
    while True:
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:  # Has data
                    # Parse latest metrics
                    latest_line = lines[-1].strip().split(',')
                    if len(latest_line) >= 8:
                        epoch = latest_line[0]
                        train_loss = latest_line[1]
                        val_loss = latest_line[2]
                        mAP50 = latest_line[6] if len(latest_line) > 6 else "N/A"
                        mAP50_95 = latest_line[7] if len(latest_line) > 7 else "N/A"
                        
                        print(f"\r🔄 Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | mAP50: {mAP50} | mAP50-95: {mAP50_95}", end="", flush=True)
                
            except Exception as e:
                print(f"\n⚠️  Error reading metrics: {e}")
        
        # Check for completed training
        if os.path.exists(os.path.join(project_dir, "weights", "best.pt")):
            print(f"\n✅ Training completed! Best model saved.")
            break
        
        time.sleep(5)  # Check every 5 seconds

def check_training_status(project_dir: str):
    """Check current training status"""
    
    if not os.path.exists(project_dir):
        print("📁 Training directory not found - training may not have started")
        return False
    
    # Check for model files
    weights_dir = os.path.join(project_dir, "weights")
    if os.path.exists(weights_dir):
        files = os.listdir(weights_dir)
        if "best.pt" in files:
            print("✅ Training completed - best model found!")
            return True
        elif "last.pt" in files:
            print("🔄 Training in progress - latest checkpoint found")
            return False
    
    # Check for results
    results_file = os.path.join(project_dir, "results.csv")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            print(f"📊 Training progress: {len(lines)-1} epochs completed")
            return False
    
    print("⏳ Training status: Initializing...")
    return False

def show_training_summary(project_dir: str):
    """Show training summary and statistics"""
    
    if not os.path.exists(project_dir):
        print("❌ No training results found")
        return
    
    print("\n📋 Training Summary:")
    print("=" * 40)
    
    # Check weights
    weights_dir = os.path.join(project_dir, "weights")
    if os.path.exists(weights_dir):
        files = os.listdir(weights_dir)
        print(f"📦 Model files: {len(files)}")
        for file in files:
            size = os.path.getsize(os.path.join(weights_dir, file)) / (1024*1024)
            print(f"   📄 {file} ({size:.1f} MB)")
    
    # Check results
    results_file = os.path.join(project_dir, "results.csv")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 1:
            print(f"📊 Training epochs: {len(lines)-1}")
            
            # Parse metrics
            headers = lines[0].strip().split(',')
            latest = lines[-1].strip().split(',')
            
            print("\n📈 Latest Metrics:")
            for i, header in enumerate(headers):
                if i < len(latest):
                    print(f"   {header}: {latest[i]}")
    
    # Check for plots
    plots = ["results.png", "confusion_matrix.png", "labels.jpg"]
    for plot in plots:
        plot_path = os.path.join(project_dir, plot)
        if os.path.exists(plot_path):
            print(f"📊 Plot available: {plot}")

def generate_training_plots(project_dir: str):
    """Generate training plots from results"""
    
    results_file = os.path.join(project_dir, "results.csv")
    if not os.path.exists(results_file):
        print("❌ No results file found")
        return
    
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Load results
        df = pd.read_csv(results_file)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO Training Progress', fontsize=16)
        
        # Loss plot
        if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Loss')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # mAP plot
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
            axes[0, 1].set_title('mAP50')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP50')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Precision plot
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall plot
        if 'metrics/recall(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(project_dir, "training_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to: {plot_path}")
        
        plt.show()
        
    except ImportError:
        print("❌ Required packages not installed. Install with: pip install pandas matplotlib")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")

def create_training_report(project_dir: str, output_path: str = None):
    """Create a comprehensive training report"""
    
    if not os.path.exists(project_dir):
        print("❌ No training results found")
        return
    
    if output_path is None:
        output_path = os.path.join(project_dir, "training_report.txt")
    
    report = []
    report.append("YOLO Training Report")
    report.append("=" * 50)
    report.append(f"Project: {project_dir}")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Model information
    weights_dir = os.path.join(project_dir, "weights")
    if os.path.exists(weights_dir):
        files = os.listdir(weights_dir)
        report.append("Model Files:")
        for file in files:
            size = os.path.getsize(os.path.join(weights_dir, file)) / (1024*1024)
            report.append(f"  - {file} ({size:.1f} MB)")
        report.append("")
    
    # Training metrics
    results_file = os.path.join(project_dir, "results.csv")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 1:
            report.append(f"Training Statistics:")
            report.append(f"  - Total epochs: {len(lines)-1}")
            
            # Parse latest metrics
            headers = lines[0].strip().split(',')
            latest = lines[-1].strip().split(',')
            
            report.append("  - Latest metrics:")
            for i, header in enumerate(headers):
                if i < len(latest):
                    report.append(f"    {header}: {latest[i]}")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Training report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Monitor")
    parser.add_argument("--project", type=str, default="training_results/exp",
                       help="Path to training project directory")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor training in real-time")
    parser.add_argument("--status", action="store_true",
                       help="Check training status")
    parser.add_argument("--summary", action="store_true",
                       help="Show training summary")
    parser.add_argument("--plots", action="store_true",
                       help="Generate training plots")
    parser.add_argument("--report", action="store_true",
                       help="Create training report")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for report")
    
    args = parser.parse_args()
    
    print("🔍 YOLO Training Monitor")
    print("=" * 40)
    
    # Check current status
    if args.status:
        check_training_status(args.project)
        return
    
    # Show summary
    if args.summary:
        show_training_summary(args.project)
        return
    
    # Generate plots
    if args.plots:
        generate_training_plots(args.project)
        return
    
    # Create report
    if args.report:
        create_training_report(args.project, args.output)
        return
    
    # Monitor training
    if args.monitor:
        try:
            monitor_training_logs(args.project)
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        
        # Show final summary
        show_training_summary(args.project)
        return
    
    # Default: show summary
    show_training_summary(args.project)

if __name__ == "__main__":
    main() 