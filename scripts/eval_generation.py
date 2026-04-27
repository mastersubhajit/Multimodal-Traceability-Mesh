import os
import torch
import json
import time
import argparse
from datasets import load_from_disk, Dataset
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from tqdm import tqdm
from peft import PeftModel
from dotenv import load_dotenv
from collections import defaultdict
import evaluate
import numpy as np

load_dotenv()

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _option_rank(generated: str, correct_label: str) -> int:
    """Return 1-based rank of correct_label among A/B/C/D found in generated text."""
    text = generated.upper()
    seen = []
    for ch in ["A", "B", "C", "D"]:
        idx = text.find(ch)
        if idx != -1:
            seen.append((idx, ch))
    seen.sort()
    labels = [ch for _, ch in seen]
    try:
        return labels.index(correct_label.upper()) + 1
    except ValueError:
        return len(labels) + 1  # not found → worst rank


def compute_citation_score(generated: str, evidence: str) -> float:
    """Unigram overlap between generated answer and evidence context."""
    if not evidence or not generated:
        return 0.0
    gen_tokens = set(generated.lower().split())
    ev_tokens = set(evidence.lower().split())
    if not gen_tokens:
        return 0.0
    return len(gen_tokens & ev_tokens) / len(gen_tokens)


def compute_ragas_metrics(results, use_ragas: bool = True):
    """
    Compute RAGAS faithfulness/answer_relevancy/context_recall/context_precision.
    Falls back to zeros when OpenAI key is unavailable or ragas fails.
    """
    if not use_ragas:
        return {}
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        from datasets import Dataset as HFDataset

        rows = [r for r in results if r.get("evidence") and r.get("expected")]
        if not rows:
            return {}

        ragas_data = HFDataset.from_dict({
            "question":    [r["question_text"] for r in rows],
            "answer":      [r["generated"] for r in rows],
            "contexts":    [[r["evidence"]] for r in rows],
            "ground_truth":[r["expected"] for r in rows],
        })
        scores = ragas_evaluate(
            ragas_data,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        return {
            "faithfulness":        float(scores["faithfulness"]),
            "answer_relevancy":    float(scores["answer_relevancy"]),
            "context_recall":      float(scores["context_recall"]),
            "context_precision":   float(scores["context_precision"]),
        }
    except Exception as e:
        print(f"[RAGAS] skipped: {e}")
        return {}


def compute_factscore(results):
    """
    Approximate FActScore: fraction of generated claims supported by evidence.
    Uses unigram precision as a lightweight stand-in (no external API needed).
    """
    scores = []
    for r in results:
        ev = r.get("evidence", "")
        gen = r.get("generated", "")
        if not ev or not gen:
            continue
        gen_tokens = gen.lower().split()
        ev_tokens = set(ev.lower().split())
        if not gen_tokens:
            continue
        precision = sum(1 for t in gen_tokens if t in ev_tokens) / len(gen_tokens)
        scores.append(precision)
    return float(np.mean(scores)) if scores else 0.0


def compute_metrics(results, rouge_metric, bleu_metric):
    predictions = [r["generated"] for r in results]
    references_rouge = [r["expected"] for r in results]
    references_bleu  = [[r["expected"]] for r in results]

    bleu_score  = bleu_metric.compute(predictions=predictions, references=references_bleu)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references_rouge)
    em = float(np.mean([
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references_rouge)
    ]))
    hits = [r for r in results if r["is_correct"]]
    accuracy = len(hits) / len(results) if results else 0.0

    # MRR — meaningful for MCQ tasks; treat as 1/rank of correct option
    mrr_scores = [r.get("mrr", 1.0 if r["is_correct"] else 0.0) for r in results]
    mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    # Hit Rate at k=1 (same as accuracy for single-answer tasks)
    hit_rate = accuracy

    # Citation score
    citation_scores = [r.get("citation_score", 0.0) for r in results]
    citation = float(np.mean(citation_scores)) if citation_scores else 0.0

    return {
        "accuracy":     accuracy,
        "hit_rate_at1": hit_rate,
        "mrr":          mrr,
        "em":           em,
        "bleu":         bleu_score["bleu"],
        "rougeL":       rouge_score["rougeL"],
        "citation":     citation,
        "total":        len(results),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_generation(
    model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    adapter_path=None,
    dataset_path="data/processed/multimodal_sft_dataset",
    output_path="logs/eval_generation_results.json",
    limit=None,
    use_ragas=False,
    use_vllm=False,
    tp_size=4,
):
    hf_token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_id}")
    vllm_engine = None
    if use_vllm:
        from src.rag.vllm_engine import VLLMTurboEngine
        vllm_engine = VLLMTurboEngine(
            model_id=model_id, 
            tensor_parallel_size=tp_size,
            kv_cache_compression=True
        )
        processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
    else:
        processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        if adapter_path:
            print(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    print(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rouge_metric = evaluate.load("rouge")
    bleu_metric  = evaluate.load("bleu")

    results = []
    latencies = []

    print("Starting inference...")
    for i, example in enumerate(tqdm(dataset)):
        full_text = example["text"]
        image = example["image"]
        
        # Parse the structured text to identify roles and content
        # We assume the format: <|system|>\n...\n<|user|>\n...\n<|assistant|>\n...
        messages = []
        try:
            if "<|system|>\n" in full_text:
                parts = full_text.split("<|system|>\n", 1)[1].split("<|user|>\n", 1)
                system_content = parts[0].strip()
                messages.append({"role": "system", "content": [{"type": "text", "text": system_content}]})
                
                user_part = parts[1].split("<|assistant|>\n", 1)
                user_content = user_part[0].strip()
                expected = user_part[1].strip() if len(user_part) > 1 else ""
            else:
                # Fallback
                user_part = full_text.split("<|user|>\n", 1)
                if len(user_part) > 1:
                    user_content = user_part[1].split("<|assistant|>\n", 1)[0].strip()
                    expected = user_part[1].split("<|assistant|>\n", 1)[1].strip() if "<|assistant|>\n" in user_part[1] else ""
                else:
                    user_content = full_text
                    expected = ""
            
            # Prepare user content with image
            clean_user_text = user_content.replace("<image>", "").replace("<|image|>", "").replace("<image 1>", "").strip()
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": clean_user_text}
                ]
            })
        except Exception as e:
            print(f"Error parsing sample {i}: {e}")
            continue

        # Use apply_chat_template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            if use_vllm:
                # Note: vLLM VQA prompt handling might differ; for now, use raw prompt
                output = vllm_engine.generate([prompt], max_tokens=100)[0]
                generated_answer = output.strip()
            else:
                output = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=False
                )
                generated_text = processor.decode(output[0], skip_special_tokens=True)
                # Extraction: Llama 3.2 decode with skip_special_tokens=True might remove headers
                # but let's be safe and try to find the assistant part
                if "assistant" in generated_text:
                    generated_answer = generated_text.split("assistant")[-1].strip()
                else:
                    # Fallback: remove input part from decoded full text
                    # We need the full decoded text including special tokens to accurately slice
                    full_decoded = processor.decode(output[0], skip_special_tokens=False)
                    if "<|start_header_id|>assistant<|end_header_id|>\n\n" in full_decoded:
                        generated_answer = full_decoded.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
                    else:
                        input_decoded = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
                        generated_answer = generated_text[len(input_decoded):].strip()
        t1 = time.time()
        latencies.append(t1 - t0)


        # Extract evidence from prompt for citation/RAGAS
        evidence = ""
        if "Document context" in prompt:
            evidence = prompt.split("Document context")[-1].split("Question:")[0].strip()
        
        # Fallback to ground truth if no explicit evidence found (common in direct VQA)
        if not evidence:
            evidence = expected

        question_text = ""
        if "Question:" in prompt:
            question_text = prompt.split("Question:")[-1].split("\n")[0].strip()
        elif "<|user|>\n" in prompt:
            # Fallback for Llama 3 formatting if "Question:" prefix missing
            question_text = prompt.split("<|user|>\n")[-1].split("<|assistant|>\n")[0].replace("<image>", "").strip()

        # Correctness
        is_correct = False
        mrr_val = 0.0
        clean_gen = generated_answer.strip().upper()
        
        if "VMCBench" in example["dataset"]:
            expected_opt = expected.replace("CORRECT_OPTION: ", "").strip().upper()
            
            # Check if expected option is explicitly mentioned as correct
            import re
            valid_patterns = [
                rf"CORRECT_OPTION:\s*{expected_opt}",
                rf"CORRECT OPTION:\s*{expected_opt}",
                rf"CORRECT ANSWER IS\s*{expected_opt}",
                rf"ANSWER IS\s*{expected_opt}",
                rf"OPTION\s*{expected_opt}\s*IS CORRECT",
                rf"^\s*{expected_opt}[\s:.)]" # "A:", "A)", "A " at start
            ]
            
            if any(re.search(p, clean_gen) for p in valid_patterns):
                is_correct = True
                mrr_val = 1.0
            else:
                rank = _option_rank(generated_answer, expected_opt)
                mrr_val = 1.0 / rank
                # If it's the first option mentioned, we count it as correct
                if rank == 1:
                    is_correct = True
        else:
            # For general VQA, check if expected answer is contained in generated
            # but using word boundaries for better precision
            import re
            pattern = re.compile(rf"\b{re.escape(expected.strip().lower())}\b")
            if pattern.search(generated_answer.lower()):
                is_correct = True
                mrr_val = 1.0
            else:
                is_correct = expected.strip().lower() in generated_answer.lower()
                mrr_val = 1.0 if is_correct else 0.0

        citation_score = compute_citation_score(generated_answer, evidence)

        results.append({
            "index":         i,
            "dataset":       example["dataset"],
            "category":      example["category"],
            "question_text": question_text,
            "expected":      expected.strip(),
            "generated":     generated_answer,
            "evidence":      evidence,
            "is_correct":    is_correct,
            "mrr":           mrr_val,
            "citation_score": citation_score,
            "latency":       t1 - t0,
        })

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    def group_metrics(data):
        if not data:
            return {}
        m = compute_metrics(data, rouge_metric, bleu_metric)
        m["factscore"] = compute_factscore(data)
        m["latency_mean"] = float(np.mean([r["latency"] for r in data]))
        return m

    report = {
        "model_id":     model_id,
        "adapter_path": adapter_path,
        "overall":      {},
        "per_dataset":  {},
        "per_category": {},
        "per_dataset_category": {},
        "ragas":        {},
        "latency": {
            "mean":  float(np.mean(latencies)),
            "std":   float(np.std(latencies)),
            "total": float(sum(latencies)),
        },
    }

    report["overall"] = group_metrics(results)

    datasets_seen = sorted(set(r["dataset"] for r in results))
    for ds in datasets_seen:
        ds_data = [r for r in results if r["dataset"] == ds]
        report["per_dataset"][ds] = group_metrics(ds_data)

        cats = sorted(set(r["category"] for r in ds_data))
        for cat in cats:
            cat_data = [r for r in ds_data if r["category"] == cat]
            key = f"{ds}/{cat}"
            report["per_category"][key] = group_metrics(cat_data)
            report["per_dataset_category"][key] = group_metrics(cat_data)

    # RAGAS (needs OpenAI key; skip silently if unavailable)
    if use_ragas:
        report["ragas"] = compute_ragas_metrics(results, use_ragas=True)

    report["details"] = results

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete → {output_path}")
    ov = report["overall"]
    print(
        f"Overall | Acc={ov.get('accuracy',0):.4f} "
        f"Hit@1={ov.get('hit_rate_at1',0):.4f} "
        f"MRR={ov.get('mrr',0):.4f} "
        f"EM={ov.get('em',0):.4f} "
        f"ROUGE-L={ov.get('rougeL',0):.4f} "
        f"BLEU={ov.get('bleu',0):.4f} "
        f"Citation={ov.get('citation',0):.4f} "
        f"FActScore={ov.get('factscore',0):.4f} "
        f"Latency={report['latency']['mean']:.2f}s"
    )
    print("\nPer-dataset/category breakdown:")
    for key, m in report["per_dataset_category"].items():
        print(
            f"  {key:<35} "
            f"Acc={m.get('accuracy',0):.3f} "
            f"MRR={m.get('mrr',0):.3f} "
            f"EM={m.get('em',0):.3f} "
            f"ROUGE-L={m.get('rougeL',0):.3f} "
            f"n={m.get('total',0)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="data/processed/multimodal_sft_dataset")
    parser.add_argument("--output_path",  type=str, default="logs/eval_generation_results.json")
    parser.add_argument("--limit",        type=int, default=None)
    parser.add_argument("--use_ragas",    action="store_true", help="Enable RAGAS metrics (requires OpenAI key)")
    parser.add_argument("--use_vllm",     action="store_true", help="Use vLLM + TurboQuant for inference")
    parser.add_argument("--tp_size",      type=int, default=4, help="Tensor Parallel size for vLLM")
    args = parser.parse_args()

    evaluate_generation(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        limit=args.limit,
        use_ragas=args.use_ragas,
        use_vllm=args.use_vllm,
        tp_size=args.tp_size
    )

