import os
import json
import argparse
from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager
from src.rag.pipeline import RAGPipeline
from tqdm import tqdm
from dotenv import load_dotenv
import evaluate

load_dotenv()

def evaluate_rag_docvqa(limit=None):
    neo4j_manager = Neo4jManager()
    
    # Get all questions from the graph
    questions_data = neo4j_manager.query(
        "MATCH (d:Document)-[:HAS_QUESTION]->(q:Question) RETURN d.id as doc_id, d.image_path as image_path, q.id as question_id, q.question_text as question, q.answers as answers"
    )
    
    if limit:
        questions_data = questions_data[:limit]
        
    print(f"Evaluating {len(questions_data)} questions using RAG Pipeline...")
    
    results = []
    processed_pipelines = {} # Cache pipelines by doc_id
    
    for item in tqdm(questions_data):
        doc_id = item["doc_id"]
        image_path = item["image_path"]
        question_text = item["question"]
        expected_answers = item["answers"]
        
        if doc_id not in processed_pipelines:
            vi = VectorIndexManager()
            try:
                vi.load(os.path.join("data/processed", f"{doc_id}"))
                pipeline = RAGPipeline(doc_id, neo4j_manager, vi, use_vision=False) # Disable vision for faster text-based evaluation
                processed_pipelines[doc_id] = pipeline
            except Exception as e:
                print(f"Error loading index for {doc_id}: {e}")
                continue
        
        pipeline = processed_pipelines[doc_id]
        
        # Handle query using RAG
        response = pipeline.handle_query(question_text, threshold=5.0) # Increase threshold for debugging
        generated_answer = response["answer"]
        
        # print scores for first result
        vi_results = pipeline.vector_index.query(question_text, top_k=1)
        if vi_results:
            print(f"Q: {question_text[:50]}... Score: {vi_results[0]['score']:.4f}")
        
        # Simple string matching for DocVQA (more flexible)
        import re
        def normalize(s):
            return re.sub(r'[^a-z0-9]', '', s.lower())
            
        generated_norm = normalize(generated_answer)
        is_correct = False
        for ans in expected_answers:
            if normalize(ans) in generated_norm:
                is_correct = True
                break
        
        results.append({
            "doc_id": doc_id,
            "question_id": item["question_id"],
            "question": question_text,
            "expected": expected_answers,
            "generated": generated_answer,
            "is_correct": is_correct,
            "is_ood": response.get("is_ood", False)
        })
        
    # Calculate accuracy
    accuracy = sum(1 for r in results if r["is_correct"]) / len(results) if results else 0
    
    report = {
        "accuracy": accuracy,
        "total": len(results),
        "details": results
    }
    
    output_path = "logs/eval_rag_docvqa_results.json"
    os.makedirs("logs", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    print(f"Report saved to {output_path}")
    
    neo4j_manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_rag_docvqa(limit=args.limit)
