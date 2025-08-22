#!/bin/bash

# Model Comparison Script
# Compares performance between specified YOLO models and creates visualizations

set -e  # Exit on any error

# Define models to compare (matching your training script)
models=("yolov8n_16bs" "yolov8s_16bs" "yolov8m_16bs" "yolo11s" "yolo11n" "yolo11m")

echo "🚀 Starting model comparison..."
echo "Models to compare: ${models[*]}"
echo ""

# Create output directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="comparison_results_${timestamp}"

echo "📁 Output directory: $output_dir"
echo ""

# Run the comparison script
python compare_models.py \
    --exp-dir "/mnt/d/exp_results/cell_track/yolo" \
    --models "cell_detection_${models[@]}" \
    --output-dir "$output_dir" \
    --create-visualizations

echo ""
echo "🎉 Comparison completed!"
echo "📊 Results saved to: $output_dir"
echo ""
echo "Generated files:"
echo "  📈 model_performance_comparison.png - Overall performance charts"
echo "  📋 metrics_table.png - Detailed metrics table"
echo "  🏆 model_ranking.png - Model ranking visualization"
echo "  📄 model_comparison_detailed.csv - Detailed results in CSV format"
echo "  🎨 *_gt_vs_pred.png - Ground truth vs predictions for each model"
echo ""
echo "💡 To view results:"
echo "   cd $output_dir && ls -la"
