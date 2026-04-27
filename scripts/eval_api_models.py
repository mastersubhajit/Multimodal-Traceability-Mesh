import os
import json
import time
import argparse
from datasets import load_from_disk
from tqdm import tqdm
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from collections import defaultdict
import evaluate
import numpy as np

load_dotenv()

def evaluate_api_models(
    model_id,
    dataset_path="data/processed/multimodal_sft_dataset",
    output_path=None,
    limit=None
):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables.")

    client = InferenceClient(model=model_id, token=hf_token)

    print(f"Loading dataset: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    results = []
    latencies = []
    
    print(f"Starting API inference for {model_id}...")
    for i, example in enumerate(tqdm(dataset)):
        full_text = example["text"]
        
        # Parse for question and expected answer
        if "<|assistant|>\n" in full_text:
            raw_prompt, expected = full_text.split("<|assistant|>\n", 1)
        else:
            raw_prompt = full_text
            expected = ""

        # Extract cleaner question text from the Llama 3 formatted prompt
        # Target the text after <|user|>\n and remove <image> tokens
        q_text = raw_prompt
        if "<|user|>\n" in raw_prompt:
            q_text = raw_prompt.split("<|user|>\n", 1)[1].strip()
        
        q_text = q_text.replace("<image>", "").replace("<|image|>", "").strip()
        
        # If it's still very long/structured, try to find the "Question:" line
        if "Question:" in q_text:
            # Take everything after Question: but stop before other major sections if they exist
            possible_q = q_text.split("Question:", 1)[1].split("\n\n")[0].strip()
            if possible_q:
                q_text = possible_q

        image = example["image"] # PIL Image

        try:
            start_time = time.time()
            # Convert PIL to bytes
            import io
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            # Attempt VQA
            response = client.visual_question_answering(image_bytes, q_text)
            end_time = time.time()
            latencies.append(end_time - start_time)

            generated_answer = response[0]['answer'] if isinstance(response, list) else response.get('answer', str(response))
        except Exception as e:
            # Fallback for text-only models
            try:
                # Use raw_prompt as the context for text-only fallback
                response = client.text_generation(raw_prompt, max_new_tokens=100)
                end_time = time.time()
                latencies.append(end_time - start_time)
                generated_answer = response.strip()
            except Exception as e2:
                print(f"API Error for index {i}: {e2}")
                generated_answer = "ERROR"
                latencies.append(0)

        is_correct = False
        clean_gen = generated_answer.strip().upper()
        if "VMCBench" in example["dataset"]:
            expected_opt = expected.replace("CORRECT_OPTION: ", "").strip().upper()
            first_char = clean_gen[0] if clean_gen else ""
            if (expected_opt == first_char 
                or f"CORRECT_OPTION: {expected_opt}" in clean_gen
                or clean_gen == expected_opt):
                is_correct = True
        else:
            import re
            pattern = re.compile(rf"\b{re.escape(expected.strip().lower())}\b")
            if pattern.search(generated_answer.lower()):
                is_correct = True
            else:
                is_correct = expected.strip().lower() in generated_answer.lower()

        results.append({
            "index": i,
            "dataset": example["dataset"],
            "category": example["category"],
            "expected": expected.strip(),
            "generated": generated_answer,
            "is_correct": is_correct,
            "latency": latencies[-1]
        })

    # Summary logic (similar to eval_generation.py)
    # ...
    report = {"overall": {"accuracy": sum(1 for r in results if r["is_correct"]) / len(results) if results else 0}}
    
    if not output_path:
        output_path = f"logs/eval_api_{model_id.replace('/', '_')}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"report": report, "details": results}, f, indent=2)

    print(f"API Evaluation complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate_api_models(args.model_id, limit=args.limit)
