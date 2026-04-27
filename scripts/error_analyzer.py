"""
Error analysis: categorizes failures from eval_generation.py or verify_answers.py output.
Produces per-category counts, error rates, and representative examples.
"""
import json
import argparse
import os
import re
from collections import defaultdict


CATEGORIES = {
    "Vision_Hallucination":  ["image", "see", "look", "visible", "shown", "appear"],
    "Retrieval_Failure":     ["no mention", "cannot find", "not found", "no information", "no context"],
    "OOD_Refusal_Fail":      [], # Model answered an OOD query instead of refusing
    "Reasoning_Error":       [],  # fallback
    "Refusal_Failure":       ["i cannot", "unable to", "not able to", "as an ai"],
    "Citation_Missing":      [],  # detected structurally
}


def categorize(result: dict) -> str:
    generated = result.get("generated", "").lower() or result.get("prediction", "").lower()
    expected  = result.get("expected",  "").lower() or result.get("ground_truth", "").lower()
    is_ood = result.get("is_ood", False)

    # Citation missing: should have cited but didn't
    if "[1]" not in generated and result.get("evidence"):
        return "Citation_Missing"

    # OOD Refusal Failure: Model answered instead of refusing, or vice-versa
    if is_ood and "no mention" not in generated:
        return "OOD_Refusal_Fail"

    for cat, keywords in CATEGORIES.items():
        if cat in ["Reasoning_Error", "OOD_Refusal_Fail", "Citation_Missing"]:
            continue
        if keywords and any(kw in generated for kw in keywords):
            return cat

    return "Reasoning_Error"


def analyze_errors(results_path: str, output_path: str = None):
    if not os.path.exists(results_path):
        print(f"File not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    # Support both eval_generation.py, verify_answers.py, and eval_rag_enhanced.py output formats
    results = (
        data.get("details")
        or data.get("per_question_results")
        or data.get("results")
        or []
    )

    if not results:
        print("No per-sample results found in JSON.")
        return

    error_counts   = defaultdict(int)
    error_examples = defaultdict(list)
    total_errors   = 0
    total          = len(results)

    for res in results:
        # Determine if this is an error based on available keys
        if "is_correct" in res:
            is_err = not res["is_correct"]
        elif "is_selected_correct" in res:
            is_err = res["is_selected_correct"] != "Yes"
        else:
            # Fallback for other potential formats
            is_err = not res.get("is_correct", True)

        if not is_err:
            continue
        total_errors += 1
        cat = categorize(res)
        error_counts[cat] += 1
        if len(error_examples[cat]) < 3:
            error_examples[cat].append({
                "question":  res.get("question_text") or res.get("question", ""),
                "expected":  res.get("expected", ""),
                "generated": res.get("generated", "")[:200],
            })

    # Per dataset/category breakdown
    dataset_errors = defaultdict(lambda: defaultdict(int))
    for res in results:
        ds  = res.get("dataset", "unknown")
        cat = res.get("category", "unknown")
        if not res.get("is_correct", False):
            dataset_errors[ds][cat] += 1

    report = {
        "total_samples":  total,
        "total_errors":   total_errors,
        "error_rate":     total_errors / total if total else 0.0,
        "categories": {
            k: {
                "count":    error_counts[k],
                "rate":     error_counts[k] / total_errors if total_errors else 0.0,
                "examples": error_examples[k],
            }
            for k in CATEGORIES
        },
        "per_dataset_category_errors": {
            ds: dict(cats) for ds, cats in dataset_errors.items()
        },
    }

    print("=== Error Analysis Report ===")
    print(f"Total samples : {total}")
    print(f"Total errors  : {total_errors}  ({report['error_rate']:.1%})")
    print()
    for cat, info in report["categories"].items():
        if info["count"] > 0:
            print(f"  {cat:<30} {info['count']:>4}  ({info['rate']:.1%})")

    print("\nPer-dataset/category error counts:")
    for ds, cats in report["per_dataset_category_errors"].items():
        for cat, cnt in cats.items():
            print(f"  {ds}/{cat:<35} {cnt} errors")

    if output_path is None:
        output_path = results_path.replace(".json", "_error_analysis.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output",  type=str, default=None)
    args = parser.parse_args()
    analyze_errors(args.results, args.output)
