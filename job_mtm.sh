#!/bin/bash
# ============================================================
# MTM Slurm Submission Script — ASL-gpu (Skynet)
# Submits MTM tasks (Fine-tuning, Ingestion, Evaluation)
# ============================================================
#SBATCH --job-name=mtm_vqa
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Exit on error
set -e

# Load necessary modules (if required by cluster)
# module load cuda/12.1

# 1. Environment Setup
echo "[MTM] Activating virtual environment..."
VENV_DIR="/home/dsai-st125998/Multimodal-Traceability-Mesh/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# 2. Check GPU Status
echo "[MTM] Checking GPU status..."
nvidia-smi

# 3. Task Selection
# Default to evaluating the system, but can be overridden by task variable
# Usage: sbatch --export=TASK=finetune job_mtm.sh
TASK=${TASK:-evaluate}

case $TASK in
    "finetune")
        echo "[MTM] Starting fine-tuning (Llama 3.2 Vision + LoRA)..."
        python models/finetune.py --config models/config.yaml
        ;;
    "evaluate")
        echo "[MTM] Starting evaluation suite..."
        python src/main.py evaluate --split test
        ;;
    "ingest")
        echo "[MTM] Starting PDF ingestion..."
        python src/main.py ingest data/sample_pdfs/*.pdf
        ;;
    "serve")
        echo "[MTM] Starting API server (production mode)..."
        # Use gunicorn or uvicorn for serving
        uvicorn src.api:app --host 0.0.0.0 --port 8000
        ;;
    *)
        echo "Unknown task: $TASK"
        exit 1
        ;;
esac

echo "[MTM] Job complete."
