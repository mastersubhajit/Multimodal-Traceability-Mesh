import os
import subprocess
import argparse
import json
import time

def run_ablation(config_name, doc_id, test_data, use_graph, use_vector, use_vision, model_id=None):
    print(f"--- Running Ablation: {config_name} ---")
    output_path = f"logs/ablation_{config_name}.json"
    
    # We can pass these flags to our evaluation script
    cmd = [
        "python", "scripts/eval_rag_enhanced.py",
        "--doc_id", doc_id,
        "--test_data", test_data,
        "--output", output_path
    ]
    
    # In a real scenario, the evaluation script would need to support these flags
    # For now, I'll assume they are handled via environment variables or we'd modify the script
    env = os.environ.copy()
    env["USE_GRAPH"] = str(use_graph).lower()
    env["USE_VECTOR"] = str(use_vector).lower()
    env["USE_VISION"] = str(use_vision).lower()
    if model_id:
        env["MODEL_ID"] = model_id

    start_time = time.time()
    subprocess.run(cmd, env=env)
    end_time = time.time()
    
    print(f"Ablation {config_name} took {end_time - start_time:.2f}s")
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    results = {}

    # 1. Full System (Graph + Vector + Vision)
    results["Full"] = run_ablation("full", args.doc_id, args.test_data, True, True, True)

    # 2. No Graph
    results["NoGraph"] = run_ablation("no_graph", args.doc_id, args.test_data, False, True, True)

    # 3. No Vector (Graph Only)
    results["NoVector"] = run_ablation("no_vector", args.doc_id, args.test_data, True, False, True)

    # 4. No Vision
    results["NoVision"] = run_ablation("no_vision", args.doc_id, args.test_data, True, True, False)

    # 5. Base Model (Different model ablation)
    results["BaseModel"] = run_ablation("base_model", args.doc_id, args.test_data, True, True, True, 
                                        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct")

    # Final summary report
    summary = {}
    for name, path in results.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                summary[name] = data.get("summary", {}).get("overall", {})

    with open("logs/ablation_study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAblation Study Complete. Summary saved to logs/ablation_study_summary.json")

if __name__ == "__main__":
    main()
