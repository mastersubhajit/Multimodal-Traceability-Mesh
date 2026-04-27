import os
import torch
import json
import argparse
from datasets import load_from_disk
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from tqdm import tqdm
from peft import PeftModel
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def evaluate_sft(
    model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    adapter_path=None,
    dataset_path="data/processed/multimodal_sft_dataset",
    output_path="logs/evaluation_report.json",
    limit=None
):
    hf_token = os.getenv("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_id}")
    processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        token=hf_token
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()

    print(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    results = []
    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    sub_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

    print("Starting inference...")
    for i, example in enumerate(tqdm(dataset)):
        full_text = example["text"]
        # Split into prompt and expected answer based on <|assistant|>
        if "<|assistant|>\n" in full_text:
            prompt, expected = full_text.split("<|assistant|>\n", 1)
            prompt += "<|assistant|>\n"
        else:
            prompt = full_text
            expected = ""

        image = example["image"]
        
        # Replace <image> with <|image|> for Llama 3.2 Vision
        prompt = prompt.replace("<image>", "<|image|>")

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
            
        generated_text = processor.decode(output[0], skip_special_tokens=True)
        # Extract assistant response
        if "assistant" in generated_text:
             generated_answer = generated_text.split("assistant")[-1].strip()
        else:
             generated_answer = generated_text[len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

        # Simple string match for VMCBench (CORRECT_OPTION: X)
        # For others, might need more complex evaluation
        is_correct = False
        if "VMCBench" in example["dataset"]:
            # Extract option from expected: "CORRECT_OPTION: A"
            expected_opt = expected.replace("CORRECT_OPTION: ", "").strip()
            # Try to find option in generated
            if f"CORRECT_OPTION: {expected_opt}" in generated_answer or generated_answer.strip() == expected_opt:
                is_correct = True
        else:
            # Fuzzy match or exact match for short answers
            is_correct = expected.strip().lower() in generated_answer.lower()

        dataset_name = example["dataset"]
        category_name = example["category"]

        metrics[dataset_name]["total"] += 1
        if is_correct:
            metrics[dataset_name]["correct"] += 1
            
        sub_metrics[f"{dataset_name}/{category_name}"]["total"] += 1
        if is_correct:
            sub_metrics[f"{dataset_name}/{category_name}"]["correct"] += 1

        results.append({
            "index": i,
            "dataset": dataset_name,
            "category": category_name,
            "prompt": prompt,
            "expected": expected,
            "generated": generated_answer,
            "is_correct": is_correct
        })

    # Summary
    report = {
        "overall": {},
        "per_dataset": {},
        "per_category": {},
        "details": results
    }

    total_correct = 0
    total_samples = 0
    for ds, m in metrics.items():
        acc = m["correct"] / m["total"] if m["total"] > 0 else 0
        report["per_dataset"][ds] = {"accuracy": acc, "correct": m["correct"], "total": m["total"]}
        total_correct += m["correct"]
        total_samples += m["total"]

    report["overall"]["accuracy"] = total_correct / total_samples if total_samples > 0 else 0
    
    for cat, m in sub_metrics.items():
        acc = m["correct"] / m["total"] if m["total"] > 0 else 0
        report["per_category"][cat] = {"accuracy": acc, "correct": m["correct"], "total": m["total"]}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation complete. Report saved to {output_path}")
    print(f"Overall Accuracy: {report['overall']['accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="data/processed/multimodal_sft_dataset")
    parser.add_argument("--output_path", type=str, default="logs/evaluation_report.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_sft(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        limit=args.limit
    )
