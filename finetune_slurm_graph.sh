#!/bin/bash
#SBATCH --job-name=mtm_finetune_graph
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_graph_%j.out
#SBATCH --error=logs/finetune_graph_%j.err

set -e

# 1. Setup Environment
echo "[MTM] Activating environment..."
VENV_DIR="/home/dsai-st125998/Multimodal-Traceability-Mesh/.venv"
source "$VENV_DIR/bin/activate"
export PYTHONPATH=$PYTHONPATH:.

# 2. Graph Dataset Formatting
DATASET_PATH="data/processed/multimodal_sft_dataset_graph"
if [ ! -d "$DATASET_PATH" ]; then
    echo "[MTM] Formatting GRAPH datasets..."
    python3 scripts/format_datasets_for_sft_graph.py --limit 100
else
    echo "[MTM] Processed dataset found at $DATASET_PATH, skipping formatting."
fi

# 3. Multi-GPU Fine-tuning using Accelerate (2 GPUs)
echo "[MTM] Starting 2-GPU GRAPH Fine-tuning on ASL-gpu using accelerate..."
accelerate launch --num_processes 2 src/vision/fine_tune_llama.py \
    --model_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --dataset_name "data/processed/multimodal_sft_dataset_graph" \
    --output_dir "models/llama-3.2-11b-vision-ft-graph"

echo "[MTM] GRAPH Fine-tuning job complete."
