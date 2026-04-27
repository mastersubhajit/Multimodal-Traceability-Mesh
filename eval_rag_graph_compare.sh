#!/bin/bash
echo "Evaluating Base Llama-3.2-11B-Vision-Instruct on Provenance Graph Data..."
.venv/bin/python3 scripts/eval_generation.py \
    --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --dataset_path data/processed/multimodal_sft_dataset_graph \
    --output_path logs/eval_graph_rag_base.json \
    --limit 10

echo "Evaluating Fine-tuned Llama-3.2-11B-Vision-Instruct (Graph Adapter) on Provenance Graph Data..."
.venv/bin/python3 scripts/eval_generation.py \
    --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --adapter_path models/llama-3.2-11b-vision-ft-graph \
    --dataset_path data/processed/multimodal_sft_dataset_graph \
    --output_path logs/eval_graph_rag_ft.json \
    --limit 10
