import argparse
import os
import hashlib
import json
from src.ingestion.parser import DocParser
from src.ingestion.mcq_detector import MCQDetector
from src.graph.neo4j_manager import Neo4jManager
from src.rag.vector_index import VectorIndexManager

def compute_doc_hash(pdf_path: str) -> str:
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main(file_path: str, clear_db: bool = False):
    # Compute doc_id (hash)
    doc_id = compute_doc_hash(file_path)
    filename = os.path.basename(file_path)

    # Initialize Managers
    neo4j_manager = Neo4jManager()
    vector_index = VectorIndexManager()

    if clear_db:
        print("Clearing Neo4j database...")
        neo4j_manager.clear_database()

    # Step 1: Parse file
    print(f"Parsing {filename}...")
    parser = DocParser()
    doc_data = parser.process_file(file_path)
    # Step 2: Detect MCQs
    print("Detecting MCQs...")
    detector = MCQDetector(doc_data)
    mcqs = detector.detect_mcqs()
    print(f"Detected {len(mcqs)} MCQs.")
    
    # Step 3: Populate Graph & Vector Index
    print(f"Populating provenance graph for {filename}...")
    neo4j_manager.create_document_node(doc_id, filename, len(doc_data["pages"]))
    
    for page in doc_data["pages"]:
        neo4j_manager.create_page_node(doc_id, page["page_no"], page["width"], page["height"])
        
    text_blocks = []
    block_ids = []
    
    for block in doc_data["blocks"]:
        neo4j_manager.create_block_node(doc_id, block["page"], block)
        if block["type"] == "text":
            import hashlib
            bbox_str = str(block.get('bbox'))
            bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
            block_id = f"{doc_id}_p{block['page']}_b_txt_{bbox_hash}"
            text_blocks.append(block)
            block_ids.append(block_id)
            
    print(f"Populating vector index with {len(text_blocks)} blocks...")
    vector_index.add_blocks(text_blocks, block_ids)
    vector_index.save(os.path.join("data/processed", f"{doc_id}"))
        
    for i, mcq in enumerate(mcqs):
        neo4j_manager.create_mcq_structure(doc_id, mcq, i)
        
    print(f"Ingestion complete. doc_id: {doc_id}")
    neo4j_manager.close()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to PDF or image file")
    parser.add_argument("--clear", action="store_true", help="Clear database before ingesting")
    args = parser.parse_args()

    if os.path.exists(args.file):
        main(args.file, clear_db=args.clear)
    else:
        print(f"File {args.file} not found.")

