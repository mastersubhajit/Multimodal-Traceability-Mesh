import os
import json
import hashlib
from datasets import load_dataset
from src.graph.neo4j_manager import Neo4jManager
from src.ingestion.parser import DocParser
from src.rag.vector_index import VectorIndexManager
from PIL import Image
from tqdm import tqdm
import argparse

def get_image_hash(image):
    # Compute hash of image pixels to use as a stable doc_id
    import hashlib
    hash_obj = hashlib.md5(image.tobytes())
    return hash_obj.hexdigest()

def ingest_dataset(name, dataset, neo4j_manager, parser, image_dir):
    print(f"Ingesting {name} ({len(dataset)} items)...")
    processed_docs = set()
    
    # We'll also collect question metadata for the evaluation script
    eval_data = []

    for item in tqdm(dataset):
        try:
            # Normalize item structure based on dataset
            if name == "DocVQA":
                doc_id = str(item['docId'])
                question = item['question']
                answers = item['answers']
                image = item['image']
                category = item['question_types'][0] if item.get('question_types') else "General"
            elif name == "TextVQA":
                doc_id = item['image_id']
                question = item['question']
                answers = item['answers']
                image = item['image']
                category = item['image_classes'][0] if item.get('image_classes') else "General"
            elif name == "VMCBench":
                # VMCBench might not have a stable doc_id in the split
                image = item['image']
                doc_id = get_image_hash(image)
                question = item['question']
                answers = [item['answer']]
                category = item['category']
            elif name == "InfographicVQA":
                # InfographicVQA from HF might have different keys
                image = item['images'][0] if isinstance(item.get('images'), list) else item.get('image')
                doc_id = get_image_hash(image)
                # InfographicVQA often has multiple user/assistant turns
                question = item['texts'][0]['user'] if isinstance(item.get('texts'), list) else item.get('question', '')
                answers = [item['texts'][0]['assistant']] if isinstance(item.get('texts'), list) else item.get('answers', [])
                category = "Infographic"
            else:
                continue

            image_path = os.path.join(image_dir, f"{name}_{doc_id}.png")
            
            if doc_id not in processed_docs:
                if not os.path.exists(image_path):
                    image.save(image_path)
                
                # Ingest into Neo4j
                neo4j_manager.create_document_node(
                    doc_id=f"{name}_{doc_id}",
                    filename=f"{name}_{doc_id}.png",
                    page_count=1,
                    image_path=image_path
                )
                
                # Parse for graph
                doc_data = parser.process_image(image_path)
                for page in doc_data["pages"]:
                    neo4j_manager.create_page_node(f"{name}_{doc_id}", page["page_no"], page["width"], page["height"])
                
                text_blocks = []
                block_ids = []
                for block in doc_data["blocks"]:
                    bbox_str = str(block.get('bbox'))
                    bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:8]
                    block_id = f"{name}_{doc_id}_p{block['page']}_b{bbox_hash}"
                    
                    neo4j_manager.create_block_node(f"{name}_{doc_id}", block["page"], block)
                    if block["type"] == "text":
                        text_blocks.append(block)
                        block_ids.append(block_id)

                if text_blocks:
                    vi = VectorIndexManager()
                    vi.add_blocks(text_blocks, block_ids)
                    vi.save(os.path.join("data/processed", f"{name}_{doc_id}"))

                processed_docs.add(doc_id)

            # Log question for eval
            eval_data.append({
                "doc_id": f"{name}_{doc_id}",
                "query": question,
                "answer": answers[0] if answers else "N/A",
                "category": f"{name}/{category}",
                "image_path": image_path,
                "dataset": name
            })
            
        except Exception as e:
            print(f"Error ingesting item from {name}: {e}")
            continue
            
    return eval_data

def main(limit=None):
    neo4j_manager = Neo4jManager()
    neo4j_manager.setup_database()
    parser = DocParser(use_ocr=True)
    image_dir = "data/processed/benchmark_images"
    os.makedirs(image_dir, exist_ok=True)
    
    total_eval_data = []

    # Datasets to process
    configs = [
        ("DocVQA", "lmms-lab/DocVQA", "DocVQA", ["validation", "test"]),
        ("TextVQA", "lmms-lab/textvqa", "default", ["validation", "test"]),
        ("VMCBench", "suyc21/VMCBench", None, ["test"]),
        ("InfographicVQA", "ayoubkirouane/infographic-VQA", None, ["train"])
    ]

    for name, path, config, splits in configs:
        for split in splits:
            print(f"Loading {name} split {split}...")
            try:
                ds = load_dataset(path, config, split=split, cache_dir="data/raw")
                if limit:
                    ds = ds.select(range(min(limit, len(ds))))
                
                eval_data = ingest_dataset(name, ds, neo4j_manager, parser, image_dir)
                total_eval_data.extend(eval_data)
            except Exception as e:
                print(f"Failed to load/ingest {name} {split}: {e}")

    # Save consolidated test data for evaluation
    with open("data/processed/total_test_data.json", "w") as f:
        json.dump(total_eval_data, f, indent=2)
    
    print(f"Consolidated eval data saved to data/processed/total_test_data.json")
    print(f"Total evaluation items: {len(total_eval_data)}")
    neo4j_manager.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.limit)
