import paddle
import argparse
import os
import hashlib
import json
from src.ingestion.parser import DocParser
from src.ingestion.mcq_detector import MCQDetector
from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager
from src.rag.pipeline import RAGPipeline
from src.vision.selection_detector import SelectionDetector

def compute_doc_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main(file_path: str, use_graph=True, use_vector=True, use_vision=True,
         model_id: str = None):
    doc_id = compute_doc_hash(file_path)
    filename = os.path.basename(file_path)

    # Initialize Managers
    neo4j_manager = Neo4jManager()
    vector_index = VectorIndexManager()
    # Attempt to load existing index
    try:
        vector_index.load(os.path.join("data/processed", f"{doc_id}"))
        print(f"Loaded existing index for {doc_id}")
    except:
        print(f"Index not found for {doc_id}. Please run scripts/ingest_to_graph.py first.")

    # Step 1: Initialize Pipeline
    parser = DocParser()
    pipeline = RAGPipeline(doc_id, neo4j_manager, vector_index,
                           llm_path=model_id,
                           use_graph=use_graph, use_vector=use_vector, use_vision=use_vision)
    
    # Share the loaded model if available
    if hasattr(pipeline, "llm") and hasattr(pipeline, "processor"):
        vision_detector = SelectionDetector(model=pipeline.llm, processor=pipeline.processor)
    else:
        vision_detector = SelectionDetector(model_path=model_id)
    
    # Step 2: Get MCQs from Graph (for now, retrieve them from detector)
    # Full implementation: Retrieve from Neo4j
    doc_data = parser.process_file(file_path)
    parser.doc_data = doc_data # Store for vision_detector
    detector = MCQDetector(doc_data)
    mcqs = detector.detect_mcqs()
    
    print(f"Verifying {len(mcqs)} MCQs in {filename}...")
    
    results = []
    for i, mcq in enumerate(mcqs):
        # 1. Detect user selection (Vision)
        mcq_with_selection = vision_detector.process_mcq_options(parser, file_path, mcq)
        
        # Determine the selected option label
        selected_label = None
        for opt in mcq_with_selection["options"]:
            if opt["is_selected"]:
                selected_label = opt["label"]
                break
        
        # 2. Verify answer (RAG)
        context = pipeline.retrieve_context(i)
        derivation = pipeline.derive_answer(context, file_path=file_path)
        
        is_correct = "N/A"
        if selected_label:
            is_correct = "Yes" if selected_label == derivation["correct_option"] else "No"
            
        verification = {
            "question_index": i,
            "page_no": mcq["page"],
            "question_bbox": mcq["bbox"],
            "question_text": mcq["question_text"],
            "detected_selection": selected_label,
            "correct_answer": derivation.get("correct_option", "N/A"),
            "is_selected_correct": is_correct,
            "reasoning": derivation.get("reasoning", derivation.get("rationale", "N/A")),
            "evidence": [{"text": b} for b in context.get("graph_context", [])]
        }
        results.append(verification)
        
        # Stage 6: Graph Update
        # Fetching evidence block IDs for graph update
        evidence_ids = [r["id"] for r in context.get("vector_evidence", [])]
        pipeline.update_graph_with_verification(i, derivation, is_correct, evidence_ids)
        
        print(f"Q{i+1}: Detected: {selected_label} | Correct: {verification['correct_answer']} | Result: {verification['is_selected_correct']}")
        
    # Step 3: Document-level Summary
    correct_count = sum(1 for r in results if r["is_selected_correct"] == "Yes")
    total_q = len(mcqs)
    
    output_data = {
        "document": filename,
        "doc_id": doc_id,
        "total_questions": total_q,
        "questions_with_selection": len([r for r in results if r["detected_selection"] is not None]),
        "score": {
            "correct": correct_count,
            "incorrect": total_q - correct_count,
            "percentage": (correct_count / total_q * 100) if total_q > 0 else 0
        },
        "per_question_results": results,
        "graph_cache_status": "EXISTING" if os.path.exists(os.path.join("data/processed", f"{doc_id}.index")) else "NEW"
    }
    
    output_path = os.path.join("data/processed", f"{filename}_verification.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full results saved to {output_path}")
    
    neo4j_manager.close()
    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to PDF or Image file")
    parser.add_argument("--no_graph",  action="store_true", help="Disable Neo4j graph context")
    parser.add_argument("--no_vector", action="store_true", help="Disable vector search context")
    parser.add_argument("--no_vision", action="store_true", help="Disable visual mesh overlay")
    parser.add_argument("--model_id",  type=str, default=None, help="Override base model path/ID")
    args = parser.parse_args()

    if os.path.exists(args.file):
        main(args.file,
             use_graph=not args.no_graph,
             use_vector=not args.no_vector,
             use_vision=not args.no_vision,
             model_id=args.model_id)
    else:
        print(f"File {args.file} not found.")
