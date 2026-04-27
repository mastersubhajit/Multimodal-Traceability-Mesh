"""
Master comprehensive evaluation script for Multimodal Traceability Mesh.
OPTIMIZED VERSION: 4-bit Quantization + Unified model loading + Sequential Processing.
"""

import os
import json
import time
import argparse
import re
import numpy as np
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import torch
from transformers import MllamaProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager
from src.eval.metrics import (
    calculate_rouge_bleu, calculate_exact_match, calculate_mcq_accuracy, calculate_mrr_mcq,
    calculate_citation_accuracy, calculate_factscore, 
    chair_score, caos_score, nope_score, ihalla_score, ood_refusal_score
)

load_dotenv()

INDEX_DIR = "data/processed"
ABSENT_OBJECTS_POOL = ["giraffe", "airplane", "snowboard", "toothbrush", "skateboard",
                       "elephant", "bicycle", "fire hydrant", "surfboard", "baseball bat"]

# ── Global Cache for Vector Indexes ──────────────────────────────────────────
VECTOR_INDEX_CACHE = {}

def load_vector_index(doc_id: str) -> VectorIndexManager:
    if doc_id in VECTOR_INDEX_CACHE:
        return VECTOR_INDEX_CACHE[doc_id]
    
    vi = VectorIndexManager()
    idx_path = os.path.join(INDEX_DIR, doc_id)
    if os.path.exists(idx_path + ".index"):
        vi.load(idx_path)
    
    VECTOR_INDEX_CACHE[doc_id] = vi
    return vi


def retrieve_context(neo4j: Neo4jManager, doc_id: str, query: str, top_k: int = 5) -> dict:
    vi = load_vector_index(doc_id)
    vector_results = vi.query(query, top_k=top_k * 2) if vi.index.ntotal > 0 else []
    retrieved_ids = [r["id"] for r in vector_results]
    blocks = neo4j.get_blocks_by_id(retrieved_ids) if retrieved_ids else []
    block_texts = [b["text"] for b in blocks if b.get("text")]
    return {
        "evidence_blocks": block_texts[:top_k],
        "evidence_ids": retrieved_ids[:top_k],
        "vector_results": vector_results[:top_k],
    }

# ── Model loading (Unified + 4-bit) ───────────────────────────────────────────

def load_vlm_unified(model_id: str, adapter_path: str = None):
    hf_token = os.getenv("HF_TOKEN")
    print(f"  Loading Unified VLM (4-bit): {model_id}")
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto", 
        token=hf_token
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"  Adding LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name="rag_adapter")
        print(f"    Adapter loaded successfully.")
    
    print(f"  Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'No device map'}")
    return model, processor


def infer(model, processor, messages: list, image: Image.Image = None,
          max_new_tokens: int = 150) -> str:
    device = next(model.parameters()).device
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    if image is not None:
        # Resize image more aggressively for evaluation speed
        if image.width > 512 or image.height > 512:
            image.thumbnail((512, 512))
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(text=prompt, return_tensors="pt").to(device)
    
    print(f"    [Infer] Input shape: {inputs['input_ids'].shape if 'input_ids' in inputs else 'N/A'}, Tokens: {inputs['input_ids'].numel() if 'input_ids' in inputs else 'N/A'}")
    
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_time = time.time() - t0
    print(f"    [Infer] Done in {gen_time:.2f}s")
    
    decoded = processor.decode(out[0], skip_special_tokens=False)
    ...


def build_base_messages(question: str, has_image: bool) -> list:
    user_content = []
    if has_image:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": f"Question: {question}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": "Answer directly."}]},
        {"role": "user", "content": user_content},
    ]


def build_rag_messages(question: str, evidence_blocks: list, has_image: bool) -> list:
    numbered = "\n".join(f"[{i+1}] {b}" for i, b in enumerate(evidence_blocks))
    user_content = []
    if has_image:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": f"Evidence:\n{numbered}\n\nQuestion: {question}"})
    return [
        {"role": "system", "content": [{"type": "text", "text": "Answer using evidence blocks only. Cite as [n]."}]},
        {"role": "user", "content": user_content},
    ]


# ── Per-item evaluation ───────────────────────────────────────────────────────

def evaluate_item_base(model, processor, item: dict) -> dict:
    img = None
    if os.path.exists(item.get("image_path", "")):
        try: img = Image.open(item["image_path"]).convert("RGB")
        except: pass

    msgs = build_base_messages(item["query"], img is not None)
    t0 = time.time()
    answer = infer(model, processor, msgs, img)
    latency = time.time() - t0

    gt = item.get("answer", "")
    is_correct = gt.lower() in answer.lower() if gt else False
    
    return {
        "doc_id": item["doc_id"], "query": item["query"], "expected": gt, "generated": answer,
        "is_correct": is_correct, "mrr": calculate_mrr_mcq([answer], [gt]),
        "latency": latency, "mode": "base"
    }


def evaluate_item_rag(model, processor, neo4j: Neo4jManager, item: dict) -> dict:
    img = None
    if os.path.exists(item.get("image_path", "")):
        try: img = Image.open(item["image_path"]).convert("RGB")
        except: pass

    t0 = time.time()
    ctx = retrieve_context(neo4j, item["doc_id"], item["query"])
    evidence_blocks = ctx["evidence_blocks"]
    
    if not evidence_blocks:
        answer = "In the document it has no mention of that."
    else:
        rag_msgs = build_rag_messages(item["query"], evidence_blocks, img is not None)
        answer = infer(model, processor, rag_msgs, img)
    latency = time.time() - t0

    gt = item.get("answer", "")
    is_correct = gt.lower() in answer.lower() if gt else False

    return {
        "doc_id": item["doc_id"], "query": item["query"], "expected": gt, "generated": answer,
        "is_correct": is_correct, "mrr": calculate_mrr_mcq([answer], [gt]),
        "latency": latency, "mode": "rag"
    }

def summarize(results: list) -> dict:
    if not results: return {}
    preds = [r["generated"] for r in results]
    gts = [r["expected"] for r in results]
    rg = calculate_rouge_bleu(preds, gts)
    return {
        "accuracy": float(np.mean([r["is_correct"] for r in results])),
        "mrr": float(np.mean([r["mrr"] for r in results])),
        "latency_mean": float(np.mean([r["latency"] for r in results])),
        "rougeL": rg["rougeL"],
        "total": len(results),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    ap.add_argument("--ft_adapter", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--output_dir", default="logs")
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    neo4j = Neo4jManager()
    
    # Load test data
    with open("data/processed/test_data_eval.json") as f:
        all_items = json.load(f)[:args.limit]
    
    print(f"Total items for evaluation: {len(all_items)}")
    
    model, processor = load_vlm_unified(args.base_model, args.ft_adapter)
    has_adapter = args.ft_adapter is not None
    
    base_results, rag_results = [], []
    
    print("\n── Processing BASE evaluation ──")
    if has_adapter:
        with model.disable_adapter():
            for item in tqdm(all_items, desc="Base Mode"):
                base_results.append(evaluate_item_base(model, processor, item))
                if len(base_results) % 5 == 0:
                    print(f"  [Base] Done {len(base_results)}/{len(all_items)}, Latency: {base_results[-1]['latency']:.2f}s")
    else:
        for item in tqdm(all_items, desc="Base Mode"):
            base_results.append(evaluate_item_base(model, processor, item))
            if len(base_results) % 5 == 0:
                print(f"  [Base] Done {len(base_results)}/{len(all_items)}, Latency: {base_results[-1]['latency']:.2f}s")

    torch.cuda.empty_cache()

    print("\n── Processing RAG evaluation ──")
    for item in tqdm(all_items, desc="RAG Mode"):
        rag_results.append(evaluate_item_rag(model, processor, neo4j, item))
        if len(rag_results) % 5 == 0:
            print(f"  [RAG] Done {len(rag_results)}/{len(all_items)}, Latency: {rag_results[-1]['latency']:.2f}s")
            
    report = {"overall": {"base": summarize(base_results), "rag": summarize(rag_results)}}
    with open(os.path.join(args.output_dir, "comprehensive_eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    print("\nEvaluation Complete.")
    print(f"Base Accuracy: {report['overall']['base']['accuracy']:.4f}")
    print(f"RAG Accuracy:  {report['overall']['rag']['accuracy']:.4f}")
    
    neo4j.close()

if __name__ == "__main__":
    main()
