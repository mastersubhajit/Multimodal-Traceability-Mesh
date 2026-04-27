"""
Ablation study: tests pipeline component combinations (graph/vector/vision)
AND compares different base models on the same verification task.
Outputs a consolidated JSON report.
"""
import os
import subprocess
import argparse
import json
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

PIPELINE_CONFIGS = [
    {"name": "full_pipeline",     "flags": []},
    {"name": "no_graph",          "flags": ["--no_graph"]},
    {"name": "no_vector",         "flags": ["--no_vector"]},
    {"name": "no_vision",         "flags": ["--no_vision"]},
    {"name": "no_graph_no_vector","flags": ["--no_graph", "--no_vector"]},
    {"name": "vanilla_vlm",       "flags": ["--no_graph", "--no_vector", "--no_vision"]},
]

BASE_MODELS = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]


def run_verify(file_path: str, extra_flags: list, model_id: str = None, timeout: int = 3600) -> dict:
    cmd = ["python3", "scripts/verify_answers.py", "--file", file_path] + extra_flags
    if model_id:
        cmd += ["--model_id", model_id]
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        elapsed = time.time() - t0
        filename = os.path.basename(file_path)
        # Using a more generic name for results if needed, but keeping consistency with verify_answers.py
        res_path = os.path.join("data/processed", f"{filename}_verification.json")
        if not os.path.exists(res_path):
            return {"error": "result file not found", "elapsed": elapsed}
        with open(res_path) as f:
            data = json.load(f)
        score = data.get("score", {})
        return {
            "correct":    score.get("correct", 0),
            "incorrect":  score.get("incorrect", 0),
            "percentage": score.get("percentage", 0.0),
            "total":      data.get("total_questions", 0),
            "elapsed":    elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed": time.time() - t0}
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t0}


def run_ablation(file_path: str, test_models: bool = True, default_model: str = None) -> dict:
    results = {"component_ablation": {}, "model_ablation": {}}

    # 1. Component ablation (default model)
    print("\n=== Component Ablation ===")
    for cfg in PIPELINE_CONFIGS:
        print(f"  Running: {cfg['name']} ...")
        res = run_verify(file_path, cfg["flags"], model_id=default_model)
        results["component_ablation"][cfg["name"]] = res
        pct = res.get("percentage", "ERR")
        print(f"    → {pct}%  | {res}")

    # 2. Model ablation (full pipeline, different base models)
    if test_models:
        print("\n=== Model Ablation ===")
        for model_id in BASE_MODELS:
            name = model_id.split("/")[-1]
            print(f"  Running: {name} ...")
            res = run_verify(file_path, [], model_id=model_id)
            results["model_ablation"][name] = res
            pct = res.get("percentage", "ERR")
            print(f"    → {pct}%  | {res}")

    # 3. Summary delta table
    baseline = results["component_ablation"].get("full_pipeline", {})
    baseline_pct = baseline.get("percentage", 0.0)
    print("\n=== Component Ablation Delta vs Full Pipeline ===")
    for name, res in results["component_ablation"].items():
        delta = res.get("percentage", 0.0) - baseline_pct
        print(f"  {name:<25} {res.get('percentage', 'N/A'):>6}%   delta={delta:+.1f}%")

    os.makedirs("logs", exist_ok=True)
    output_path = "logs/ablation_study.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation study saved → {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to PDF or Image file")
    parser.add_argument("--model_id", type=str, default=None, help="Default model for component ablation")
    parser.add_argument("--no_model_ablation", action="store_true",
                        help="Skip testing different base models (faster)")
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        run_ablation(args.file, test_models=not args.no_model_ablation, default_model=args.model_id)
    else:
        print(f"File {args.file} not found.")
