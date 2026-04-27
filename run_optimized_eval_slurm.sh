#!/bin/bash
#SBATCH --job-name=mtm_optimized_eval
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/full_eval_optimized_%j.out
#SBATCH --error=logs/full_eval_optimized_%j.err

set -e

WORKDIR="/home/dsai-st125998/Multimodal-Traceability-Mesh"
cd "$WORKDIR"

source .venv/bin/activate
export PYTHONPATH="$WORKDIR:$PYTHONPATH"

if [ -f .env ]; then
    set -o allexport; source .env; set +o allexport
fi

mkdir -p logs

BASE_MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
FT_ADAPTER="models/llama-3.2-11b-vision-ft"
GRAPH_ADAPTER="models/llama-3.2-11b-vision-ft-graph"

# DocVQA doc_ids available in the graph
DOC_IDS="DocVQA_5185 DocVQA_6982 DocVQA_9419 DocVQA_4779 DocVQA_4806"

echo "============================================================"
echo " MTM Optimized Evaluation (Target: < 1hr) — $(date)"
echo " Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# Step 1: Ingest (Quick check)
if [ ! -f "data/processed/test_data_eval.json" ]; then
    echo "[1/3] Ingesting test datasets..."
    python scripts/ingest_test_val_datasets.py --limit 100
fi

# Step 2: Optimized Comprehensive Eval
# Using limit 100 per split = 200 total items
echo "[2/3] Running optimized comprehensive evaluation..."
python scripts/run_comprehensive_eval.py \
    --base_model         "$BASE_MODEL" \
    --ft_adapter         "$FT_ADAPTER" \
    --limit              100 \
    --output_dir         logs

# Step 3: Graph Eval
echo "[3/3] Running optimized graph evaluation..."
python scripts/eval_graph_ft_comparison.py \
    --doc_ids    $DOC_IDS \
    --base_model "$BASE_MODEL" \
    --ft_adapter "$GRAPH_ADAPTER" \
    --output     logs/graph_ft_comparison_optimized.json

echo "============================================================"
echo " Optimized evaluation complete — $(date)"
echo "============================================================"
