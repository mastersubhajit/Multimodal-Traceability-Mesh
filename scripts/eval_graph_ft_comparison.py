"""
Graph Comprehension Evaluation: Unified version.
Optimized to load the model once and use togglable adapters.
"""
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from peft import PeftModel

from src.graph.neo4j_manager import Neo4jManager
from src.eval.metrics import calculate_graph_type_accuracy, GRAPH_TASK_TYPES

load_dotenv()

def load_vlm_unified(model_id: str, adapter_path: str = None):
    hf_token = os.getenv("HF_TOKEN")
    processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token
    )
    if adapter_path and os.path.exists(adapter_path):
        print(f"  Adding LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name="graph_adapter")
    return model.eval(), processor

def run_inference(model, processor, prompt: str, max_new_tokens: int = 80) -> str:
    device = next(model.parameters()).device
    inputs = processor(text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = processor.decode(out[0], skip_special_tokens=False)
    if "<|start_header_id|>assistant<|end_header_id|>\n\n" in decoded:
        return decoded.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1] \
                      .replace("<|eot_id|>", "").strip()
    return processor.decode(out[0], skip_special_tokens=True).strip()

def generate_graph_questions(neo4j: Neo4jManager, doc_id: str) -> list:
    questions = []
    node_q = neo4j.query("MATCH (n {doc_id: $doc_id}) RETURN count(n) AS c", {"doc_id": doc_id})
    questions.append({"type": "N. number", "question": f"Node count for {doc_id}?", "answer": str(node_q[0]["c"]) if node_q else "0"})
    edge_q = neo4j.query("MATCH (n {doc_id: $doc_id})-[r]->() RETURN count(r) AS c", {"doc_id": doc_id})
    questions.append({"type": "E. number", "question": f"Edge count for {doc_id}?", "answer": str(edge_q[0]["c"]) if edge_q else "0"})
    return questions

def evaluate_model_on_graph(model, processor, neo4j: Neo4jManager, doc_ids: list, label: str) -> list:
    all_results = []
    for doc_id in doc_ids:
        questions = generate_graph_questions(neo4j, doc_id)
        for q in tqdm(questions, desc=f"[{label}] {doc_id}"):
            msgs = [{"role": "user", "content": [{"type": "text", "text": q["question"]}]}]
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
            generated = run_inference(model, processor, prompt)
            all_results.append({"doc_id": doc_id, "type": q["type"], "expected": q["answer"], "generated": generated, "is_correct": q["answer"] in generated})
    return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_ids", nargs="+", required=True)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--ft_adapter", default=None)
    parser.add_argument("--output", default="logs/graph_ft_comparison.json")
    args = parser.parse_args()
    neo4j = Neo4jManager()
    model, processor = load_vlm_unified(args.base_model, args.ft_adapter)
    
    print("Evaluating Base...")
    if args.ft_adapter:
        with model.disable_adapter():
            base_results = evaluate_model_on_graph(model, processor, neo4j, args.doc_ids, "base")
    else:
        base_results = evaluate_model_on_graph(model, processor, neo4j, args.doc_ids, "base")
        
    print("Evaluating Fine-tuned...")
    ft_results = evaluate_model_on_graph(model, processor, neo4j, args.doc_ids, "finetuned")
    
    report = {"base": base_results, "finetuned": ft_results}
    with open(args.output, "w") as f: json.dump(report, f, indent=2)
    neo4j.close()

if __name__ == "__main__":
    main()
