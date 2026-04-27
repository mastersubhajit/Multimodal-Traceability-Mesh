#!/bin/bash
#SBATCH --job-name=mtm_total_eval
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=120:00:00
#SBATCH --output=logs/total_eval_%j.out
#SBATCH --error=logs/total_eval_%j.err

set -e

# 1. Environment
echo "[MTM-Total] Activating environment..."
VENV_DIR="/home/dsai-st125998/Multimodal-Traceability-Mesh/.venv"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$(pwd):$PYTHONPATH"

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

mkdir -p logs data/processed/benchmark_images

# 2. Ingestion Phase
echo "[MTM-Total] Step 1/3: Ingesting total benchmarks into Neo4j and FAISS..."
# Limit removed to process total dataset
python3 scripts/ingest_total_benchmarks.py

# 3. RAG Evaluation Phase
echo "[MTM-Total] Step 2/3: Running comprehensive RAG evaluation..."
python3 scripts/eval_rag_enhanced.py \
    --test_data data/processed/total_test_data.json \
    --output logs/eval_rag_total_full.json

# 4. Graph Tasks Phase
echo "[MTM-Total] Step 3/3: Running graph comprehension tasks..."
# Example run on a representative doc_id
python3 scripts/eval_graph.py --doc_id DocVQA_11190 --output_path logs/eval_graph_total_final.json

echo "[MTM-Total] Full evaluation pipeline complete."
