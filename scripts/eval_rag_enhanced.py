import os
import json
import time
import argparse
import numpy as np
from src.rag.pipeline import RAGPipeline
from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager
from src.eval.metrics import calculate_recall, calculate_mrr, calculate_hit_rate, calculate_exact_match
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not available, skipping RAGAS metrics.")

load_dotenv()

def run_enhanced_evaluation(test_data_path, output_path):
    neo4j_manager = Neo4jManager()
    vector_index_manager = VectorIndexManager()
    
    # Cache for pipelines to avoid reloading models if they share them
    pipelines = {}

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    results = []
    ragas_data = []

    # Respect environment variables for ablation studies
    use_graph = os.getenv("USE_GRAPH", "true").lower() == "true"
    use_vector = os.getenv("USE_VECTOR", "true").lower() == "true"
    use_vision = os.getenv("USE_VISION", "true").lower() == "true"
    model_id = os.getenv("MODEL_ID", None)

    print(f"Evaluating RAG for {len(test_data)} items...")
    for item in tqdm(test_data):
        doc_id = item.get("doc_id")
        query = item["query"]
        ground_truth_answer = item.get("answer", "")
        ground_truth_ids = item.get("ground_truth_ids", [])
        category = item.get("category", "General")
        image_path = item.get("image_path")

        if not doc_id:
            print(f"Skipping item with no doc_id: {query}")
            continue

        # Get or create pipeline for this doc_id
        if doc_id not in pipelines:
            try:
                # We reuse the same model if model_id is constant, but RAGPipeline 
                # currently reloads everything. In a production eval, we'd pass 
                # the model instance. 
                # For now, let's assume we can load it. 
                # OPTIMIZATION: In a real run, I'd refactor RAGPipeline to accept an existing model.
                pipelines[doc_id] = RAGPipeline(
                    doc_id, 
                    neo4j_manager, 
                    vector_index_manager, 
                    llm_path=model_id,
                    use_graph=use_graph,
                    use_vector=use_vector,
                    use_vision=use_vision
                )
            except Exception as e:
                print(f"Failed to initialize pipeline for {doc_id}: {e}")
                continue

        pipeline = pipelines[doc_id]
        start_time = time.time()
        
        # Let's use handle_query for open-ended or derive_answer for MCQs
        dataset_name = item.get("dataset", "")
        if "options" in item or dataset_name == "VMCBench": 
            # For VMCBench we might need to synthesize context or use graph
            # Current logic: retrieve_context uses q_index which implies fixed MCQs in graph
            # If it's a general query, we use handle_query
            response = pipeline.handle_query(query)
            prediction = response.get("answer", "")
            reasoning = response.get("reasoning", "")
            is_ood = response.get("is_ood", False)
            retrieved_ids = response.get("evidence_ids", [])
            evidence_texts = [prediction] 
        else:
            # Open-ended Evaluation
            response = pipeline.handle_query(query)
            prediction = response.get("answer", "")
            reasoning = response.get("reasoning", "")
            is_ood = response.get("is_ood", False)
            retrieved_ids = response.get("evidence_ids", [])
            evidence_texts = [prediction] 

        latency = time.time() - start_time

        # Metrics
        recall = calculate_recall(retrieved_ids, ground_truth_ids)
        mrr = calculate_mrr(retrieved_ids, ground_truth_ids)
        hit_rate = calculate_hit_rate(retrieved_ids, ground_truth_ids)
        em = calculate_exact_match(prediction, ground_truth_answer)

        res_item = {
            "query": query,
            "category": category,
            "prediction": prediction,
            "ground_truth": ground_truth_answer,
            "is_ood": is_ood,
            "latency": latency,
            "recall": recall,
            "mrr": mrr,
            "hit_rate": hit_rate,
            "exact_match": em,
            "reasoning": reasoning
        }
        results.append(res_item)

        if RAGAS_AVAILABLE:
            ragas_data.append({
                "question": query,
                "answer": prediction,
                "contexts": evidence_texts,
                "ground_truth": ground_truth_answer
            })

    # RAGAS Evaluation
    ragas_results = {}
    if RAGAS_AVAILABLE and ragas_data:
        dataset = Dataset.from_list(ragas_data)
        ragas_results = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevance, context_precision, context_recall]
        )

    # Summary and Comparisons
    summary = {
        "overall": {
            "recall": np.mean([r["recall"] for r in results]),
            "mrr": np.mean([r["mrr"] for r in results]),
            "hit_rate": np.mean([r["hit_rate"] for r in results]),
            "accuracy": np.mean([r["exact_match"] for r in results]),
            "avg_latency": np.mean([r["latency"] for r in results]),
        },
        "ragas": dict(ragas_results) if ragas_results else {},
        "by_category": {}
    }

    categories = set(r["category"] for r in results)
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        summary["by_category"][cat] = {
            "recall": np.mean([r["recall"] for r in cat_items]),
            "accuracy": np.mean([r["exact_match"] for r in cat_items]),
        }

    final_report = {
        "summary": summary,
        "details": results
    }

    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, default="logs/eval_rag_enhanced.json")
    args = parser.parse_args()
    
    run_enhanced_evaluation(args.test_data, args.output)
