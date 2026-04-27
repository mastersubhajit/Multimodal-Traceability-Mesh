import os
import json
import time
import argparse
import numpy as np
from src.rag.pipeline import RAGPipeline
from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def evaluate_retrieval(doc_id, queries, ground_truth_ids):
    """
    queries: list of query strings
    ground_truth_ids: list of expected node IDs or content hashes
    """
    neo4j_manager = Neo4jManager()
    vector_index = VectorIndexManager()
    
    try:
        vector_index.load(os.path.join("data/processed", f"{doc_id}"))
    except:
        print(f"Index not found for {doc_id}")
        return {}

    pipeline = RAGPipeline(doc_id, neo4j_manager, vector_index)
    
    hits = 0
    rr_sum = 0
    latencies = []
    
    results = []
    
    print(f"Evaluating retrieval for {len(queries)} queries...")
    for q, gt_id in tqdm(zip(queries, ground_truth_ids), total=len(queries)):
        start_time = time.time()
        context = pipeline.retrieve_context(q) # Adjusted to take string or index
        end_time = time.time()
        latencies.append(end_time - start_time)
        
        # Check if GT ID is in retrieved evidence
        retrieved_ids = [r["id"] for r in context.get("vector_evidence", [])]
        
        hit = 0
        rr = 0
        if gt_id in retrieved_ids:
            hit = 1
            rank = retrieved_ids.index(gt_id) + 1
            rr = 1.0 / rank
            
        hits += hit
        rr_sum += rr
        
        results.append({
            "query": q,
            "hit": hit,
            "mrr": rr,
            "retrieved_count": len(retrieved_ids)
        })
        
    metrics = {
        "hit_rate": hits / len(queries) if queries else 0,
        "mrr": rr_sum / len(queries) if queries else 0,
        "avg_latency": np.mean(latencies),
        "details": results
    }
    
    neo4j_manager.close()
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id", type=str, required=True)
    parser.add_argument("--test_file", type=str, help="JSON file with queries and GT IDs")
    args = parser.parse_args()
    
    # Example logic: load from a generated test set
    if args.test_file and os.path.exists(args.test_file):
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
        
        queries = [item["query"] for item in test_data]
        gt_ids = [item["gt_id"] for item in test_data]
        
        metrics = evaluate_retrieval(args.doc_id, queries, gt_ids)
        print(json.dumps(metrics, indent=2))
    else:
        print("Please provide a test_file with ground truth.")
