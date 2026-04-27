#!/bin/bash
#SBATCH --job-name=mtm_comprehensive_eval
#SBATCH --partition=ASL-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=96:00:00
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err

set -e

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
echo "[MTM-Eval] Activating environment..."
VENV_DIR="/home/dsai-st125998/Multimodal-Traceability-Mesh/.venv"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Enable TurboQuant performance features
export TQ_MODE="hybrid"

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

BASE_MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
ADAPTER_PATH="models/llama-3.2-11b-vision-ft-graph"
DOC_ID="DocVQA_499dc74d"
SAMPLE_IMAGE="data/processed/docvqa_images/14465.png"
DATASET_PATH="data/processed/multimodal_sft_dataset_graph"

mkdir -p logs

# ---------------------------------------------------------------------------
# 1. Generation Evaluation — Base Model
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 1/8  Generation eval — Base Model..."
python3 scripts/eval_generation.py \
    --model_id  "$BASE_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --output_path "logs/eval_base_results.json" \
    --limit 20

# ---------------------------------------------------------------------------
# 2. Generation Evaluation — Fine-tuned Model
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 2/8  Generation eval — Fine-tuned Model..."
if [ -d "$ADAPTER_PATH" ]; then
    python3 scripts/eval_generation.py \
        --model_id    "$BASE_MODEL" \
        --adapter_path "$ADAPTER_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_path  "logs/eval_ft_results.json" \
        --limit 20
else
    echo "[MTM-Eval] WARNING: Adapter not found at $ADAPTER_PATH. Skipping FT generation eval."
fi

# ---------------------------------------------------------------------------
# 3. Graph Comprehension Evaluation
#    Metrics: N.number, E.number, Triple listing, N.degree, Highest N.degree,
#             N.description — Base vs Fine-tuned
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 3/8  Graph comprehension — Base..."
python3 scripts/eval_graph.py \
    --doc_id      "$DOC_ID" \
    --output_path "logs/eval_graph_base.json"

echo "[MTM-Eval] 3/8  Graph comprehension — Fine-tuned..."
if [ -d "$ADAPTER_PATH" ]; then
    python3 scripts/eval_graph.py \
        --doc_id      "$DOC_ID" \
        --adapter     "$ADAPTER_PATH" \
        --output_path "logs/eval_graph_ft.json"
fi

# ---------------------------------------------------------------------------
# 4. Vision Hallucination Evaluation
#    Metrics: CHAIR-i, CHAIR-s, CAOS, NOPE accuracy, I-HallA accuracy,
#             OOD refusal rate, OOD reasoning rate, ROUGE-L, BLEU
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 4/8  Vision hallucination eval..."
python3 scripts/eval_vision.py \
    --model_id    "$BASE_MODEL" \
    --image_dir   "data/processed/docvqa_images" \
    --output_path "logs/eval_vision_base.json" \
    --limit       20

if [ -d "$ADAPTER_PATH" ]; then
    python3 scripts/eval_vision.py \
        --model_id    "$BASE_MODEL" \
        --adapter_path "$ADAPTER_PATH" \
        --image_dir   "data/processed/docvqa_images" \
        --output_path "logs/eval_vision_ft.json" \
        --limit       20
fi

# ---------------------------------------------------------------------------
# 5. API/Comparative Model Evaluation (Qwen2-VL-7B via HF Inference API)
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 5/8  API comparative eval — SKIPPED (No API allowed)"
# python3 scripts/eval_api_models.py \
#     --model_id "Qwen/Qwen2-VL-7B-Instruct" \
#     --limit    100

# ---------------------------------------------------------------------------
# 6. Ablation Study
#    Component ablation: full / no_graph / no_vector / no_vision /
#                        no_graph+no_vector / vanilla_vlm
#    Model ablation:     Llama-3.2-11B vs Qwen2-VL-7B
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 6/8  Ablation study..."
if [ -f "$SAMPLE_IMAGE" ]; then
    python3 scripts/run_ablations.py \
        --file "$SAMPLE_IMAGE" \
        --model_id "$BASE_MODEL"
else
    echo "[MTM-Eval] WARNING: $SAMPLE_IMAGE not found. Skipping ablation study."
fi

# ---------------------------------------------------------------------------
# 7. Error Analysis
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 7/8  Error analysis..."
for RESULT_FILE in logs/eval_ft_results.json logs/eval_base_results.json; do
    if [ -f "$RESULT_FILE" ]; then
        python3 scripts/error_analyzer.py --results "$RESULT_FILE"
    fi
done

# ---------------------------------------------------------------------------
# 8. TruLens Evaluation (using NEBIUS)
# ---------------------------------------------------------------------------
echo "[MTM-Eval] 8/8  TruLens evaluation..."
if [ -f "logs/eval_base_results.json" ]; then
    python3 scripts/evaluate_trulens.py --results "logs/eval_base_results.json"
fi

if [ -f "logs/eval_ft_results.json" ]; then
    python3 scripts/evaluate_trulens.py --results "logs/eval_ft_results.json"
fi

echo "[MTM-Eval] All evaluations complete."
