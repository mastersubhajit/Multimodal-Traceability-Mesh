#!/bin/bash
#SBATCH --job-name=mtm_eval_local
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_local_%j.out
#SBATCH --error=logs/eval_local_%j.err

# Activate environment
source .venv/bin/activate

# Set HF Token for model access
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d '=' -f2)

echo "Starting local evaluation with Llama-3.3-70B-Instruct on 2 GPUs..."

# Run evaluation on fine-tuned results
python scripts/evaluate_trulens_local.py --results logs/eval_ft_results.json

# Run evaluation on base results
python scripts/evaluate_trulens_local.py --results logs/eval_base_results.json

echo "Evaluation finished."
