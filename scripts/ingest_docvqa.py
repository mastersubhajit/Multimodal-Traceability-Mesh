import os
import json
from datasets import load_from_disk
from src.graph.neo4j_manager import Neo4jManager
from src.ingestion.parser import DocParser
from PIL import Image
import argparse
from tqdm import tqdm

from src.rag.vector_index import VectorIndexManager
import hashlib

def main(dataset_path: str, limit: int = None, clear: bool = False):
    # Initialize Neo4j
    neo4j_manager = Neo4jManager()
    if clear:
        neo4j_manager.clear_database()
    neo4j_manager.setup_database()
    
    # Initialize Vector Index Manager
    vector_index = VectorIndexManager()

    # Initialize DocParser (ENABLE OCR to get text blocks)
    parser = DocParser(use_ocr=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        from datasets import load_dataset
        # Load arrow files
        data_files = {
            "validation": os.path.join(dataset_path, "doc_vqa-validation-*.arrow"),
            "test": os.path.join(dataset_path, "doc_vqa-test-*.arrow")
        }
        dataset = load_dataset("arrow", data_files=data_files)
    except Exception as e:
        print(f"Error loading with load_dataset: {e}")
        return

    # We'll use the 'validation' split if available, otherwise just the dataset itself
    if isinstance(dataset, dict):
        if 'validation' in dataset:
            ds = dataset['validation']
        else:
            ds = dataset[list(dataset.keys())[0]]
    else:
        ds = dataset

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # Create directory for images
    image_dir = "data/processed/docvqa_images"
    os.makedirs(image_dir, exist_ok=True)

    print(f"Ingesting {len(ds)} items...")
    
    # Track processed documents to avoid redundant node creation and image saving
    processed_docs = set()

    for item in tqdm(ds):
        doc_id = str(item['docId'])
        question_id = item['questionId']
        question_text = item['question']
        answers = item['answers']
        image = item['image'] # PIL Image

        # Save image and create document node if not already done
        image_path = os.path.join(image_dir, f"{doc_id}.png")
        if doc_id not in processed_docs:
            if not os.path.exists(image_path):
                image.save(image_path)
            
            # Step 1: Create Document node
            neo4j_manager.create_document_node(
                doc_id=doc_id,
                filename=f"{doc_id}.png",
                page_count=1,
                image_path=image_path
            )
            
            # Step 2: Parse image for provenance graph (blocks, pages)
            doc_data = parser.process_image(image_path)
            
            # Create Page node
            for page in doc_data["pages"]:
                neo4j_manager.create_page_node(doc_id, page["page_no"], page["width"], page["height"])
                
            # Create Block nodes and Vector Index
            text_blocks = []
            block_ids = []
            for block in doc_data["blocks"]:
                # Generate block_id consistent with Neo4jManager
                bbox_str = str(block.get('bbox'))
                bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
                block_id = f"{doc_id}_p{block['page']}_b{block.get('image_id', 'txt')}_{bbox_hash}"
                
                neo4j_manager.create_block_node(doc_id, block["page"], block)
                
                if block["type"] == "text":
                    text_blocks.append(block)
                    block_ids.append(block_id)

            if text_blocks:
                doc_vector_index = VectorIndexManager()
                doc_vector_index.add_blocks(text_blocks, block_ids)
                doc_vector_index.save(os.path.join("data/processed", f"{doc_id}"))

            processed_docs.add(doc_id)

        # Create Question node and link to Document
        neo4j_manager.create_docvqa_question_node(
            doc_id=doc_id,
            question_id=question_id,
            question_text=question_text,
            answers=answers
        )

    print("Ingestion complete.")
    neo4j_manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/raw/lmms-lab___doc_vqa/DocVQA/0.0.0/539088ef8a8ada01ac8e2e6d4e372586748a265e")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to ingest")
    parser.add_argument("--clear", action="store_true", help="Clear database before ingestion")
    args = parser.parse_args()
    
    main(args.dataset_path, limit=args.limit, clear=args.clear)
