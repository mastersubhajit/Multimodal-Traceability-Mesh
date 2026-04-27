#!/bin/bash
#SBATCH --job-name=mtm_finetune_full
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_full_%j.out
#SBATCH --error=logs/finetune_full_%j.err

set -e

# 1. Setup Environment
echo "[MTM] Activating environment..."
VENV_DIR="/home/dsai-st125998/Multimodal-Traceability-Mesh/.venv"
source "$VENV_DIR/bin/activate"

# 2. Full Dataset Formatting
DATASET_PATH="data/processed/multimodal_sft_dataset"
if [ ! -d "$DATASET_PATH" ]; then
    echo "[MTM] Formatting FULL datasets..."
    # Limit removed to include ALL samples from all datasets
    python3 scripts/format_datasets_for_sft.py
else
    echo "[MTM] Processed dataset found at $DATASET_PATH, skipping formatting."
fi

# 3. Distributed Fine-tuning (1 GPU on ASL-gpu)
echo "[MTM] Starting 1-GPU FULL Fine-tuning on ASL-gpu..."
# We don't need torchrun for 1 GPU, but we can still use it or just call python
python3 src/vision/fine_tune_llama.py \
    --model_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --dataset_name "data/processed/multimodal_sft_dataset" \
    --output_dir "models/llama-3.2-11b-vision-ft"

echo "[MTM] FULL Fine-tuning job complete."
