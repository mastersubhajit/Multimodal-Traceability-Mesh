#!/bin/bash
#SBATCH --job-name=mtm_full_eval
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=logs/full_eval_%j.out
#SBATCH --error=logs/full_eval_%j.err

set -e

# ── Environment ───────────────────────────────────────────────────────────────
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

# DocVQA doc_ids available in the graph (from validation split ingestion)
DOC_IDS="DocVQA_5185 DocVQA_6982 DocVQA_9419 DocVQA_4779 DocVQA_4806"

echo "============================================================"
echo " MTM Full Evaluation — $(date)"
echo " Node: $SLURMD_NODENAME  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── Step 1: Ingest (skip if data already exists) ─────────────────────────────
if [ -f "data/processed/test_data_eval.json" ] && [ -f "data/processed/val_data_eval.json" ]; then
    echo "[1/3] Ingestion data found — skipping ingest step."
else
    echo "[1/3] Ingesting test/val datasets into Neo4j (no answers stored)..."
    python scripts/ingest_test_val_datasets.py --limit 200
fi

# ── Step 2: Comprehensive RAG + Hallucination + OOD + CSQA/OBQA Evaluation ───
echo ""
echo "[2/3] Running comprehensive evaluation (base vs RAG-adapted)..."
python scripts/run_comprehensive_eval.py \
    --base_model         "$BASE_MODEL" \
    --ft_adapter         "$FT_ADAPTER" \
    --test_data          data/processed/test_data_eval.json \
    --val_data           data/processed/val_data_eval.json \
    --splits             test val \
    --limit              100 \
    --hallucination_limit 20 \
    --ood_limit          20 \
    --csqa_limit         100 \
    --obqa_limit         100 \
    --output_dir         logs

echo "[2/3] Comprehensive eval complete."

# ── Step 3: Graph comprehension — base vs fine-tuned ─────────────────────────
echo ""
echo "[3/3] Running graph comprehension evaluation (before/after finetuning)..."
if [ -d "$GRAPH_ADAPTER" ]; then
    python scripts/eval_graph_ft_comparison.py \
        --doc_ids    $DOC_IDS \
        --base_model "$BASE_MODEL" \
        --ft_adapter "$GRAPH_ADAPTER" \
        --output     logs/graph_ft_comparison.json
else
    echo "  Graph adapter not found at $GRAPH_ADAPTER — running base only."
    python scripts/eval_graph_ft_comparison.py \
        --doc_ids    $DOC_IDS \
        --base_model "$BASE_MODEL" \
        --output     logs/graph_ft_comparison.json
fi

echo "[3/3] Graph eval complete."

echo ""
echo "============================================================"
echo " All evaluations done — $(date)"
echo " Reports:"
echo "   logs/comprehensive_eval_report.json"
echo "   logs/comprehensive_eval_summary.txt"
echo "   logs/graph_ft_comparison.json"
echo "============================================================"
