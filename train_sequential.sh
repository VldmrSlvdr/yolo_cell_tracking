#!/bin/bash

# Sequential YOLO Training Script
# Trains YOLO models one by one: yolov8n, yolov8s, yolov8m, yolo11n, yolo11s, yolo11m

set -e  # Exit on any error

# Define models to train
models=("yolov8n" "yolov8s" "yolov8m" "yolo11n" "yolo11s" "yolo11m")

echo "🚀 Starting sequential training for ${#models[@]} models..."
echo "Models to train: ${models[*]}"
echo ""

# Track training results
successful_trains=0
failed_trains=0
start_time=$(date)

# Train each model sequentially
for model in "${models[@]}"; do
    echo "="*60
    echo "🔄 Starting training for: $model"
    echo "="*60
    echo "⏰ Start time: $(date)"
    
    # Run training command
    if python train_all_models.py --model "$model"; then
        echo "✅ Successfully completed training for: $model"
        ((successful_trains++))
    else
        echo "❌ Training failed for: $model"
        ((failed_trains++))
        echo "⚠️  Continuing with next model..."
    fi
    
    echo "⏰ End time: $(date)"
    echo ""
done

# Print final summary
echo "="*60
echo "🏁 TRAINING SUMMARY"
echo "="*60
echo "📅 Started: $start_time"
echo "📅 Finished: $(date)"
echo "✅ Successful: $successful_trains models"
echo "❌ Failed: $failed_trains models"
echo "📊 Success rate: $((successful_trains * 100 / ${#models[@]}))%"
echo ""
echo "Trained models: ${models[*]}"
echo "="*60

# Deactivate environment (optional)
echo "🔄 Training script completed!"
