"""
Ingest test and validation dataset splits into Neo4j WITHOUT storing correct answers.

Graph structure created:
  Document → Page → Block (text/image)
  Document → Question  (question_text only, NO answer field)

Correct answers are saved ONLY to local JSON files for offline evaluation:
  data/processed/test_data_eval.json
  data/processed/val_data_eval.json

The RAG-adapted model retrieves context (blocks) from the graph during evaluation.
The base model receives no graph context.
"""
import os
import json
import hashlib
import argparse
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

from src.graph.neo4j_manager import Neo4jManager
from src.ingestion.parser import DocParser
from src.rag.vector_index import VectorIndexManager

load_dotenv()

IMAGE_DIR = "data/processed/benchmark_images"
INDEX_DIR = "data/processed"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


def image_hash(image: Image.Image) -> str:
    return hashlib.md5(image.tobytes()).hexdigest()


def norm_item(name: str, item: dict) -> dict:
    """Normalize heterogeneous dataset item into a common schema."""
    if name == "DocVQA":
        return {
            "raw_doc_id": str(item["docId"]),
            "question":   item["question"],
            "answers":    list(item.get("answers", [])),
            "image":      item["image"],
            "category":   (item.get("question_types") or ["General"])[0],
        }
    if name == "TextVQA":
        return {
            "raw_doc_id": str(item["image_id"]),
            "question":   item["question"],
            "answers":    list(item.get("answers", [])),
            "image":      item["image"],
            "category":   (item.get("image_classes") or ["General"])[0],
        }
    if name == "VMCBench":
        return {
            "raw_doc_id": image_hash(item["image"]),
            "question":   item["question"],
            "answers":    [item["answer"]],
            "image":      item["image"],
            "category":   item.get("category", "General"),
        }
    if name == "InfographicVQA":
        img = item["images"][0] if isinstance(item.get("images"), list) else item.get("image")
        if isinstance(item.get("texts"), list) and item["texts"]:
            q = item["texts"][0].get("user", "")
            a = [item["texts"][0].get("assistant", "")]
        else:
            q = item.get("question", "")
            a = item.get("answers", [])
        return {
            "raw_doc_id": image_hash(img) if img else "unknown",
            "question":   q,
            "answers":    a,
            "image":      img,
            "category":   "Infographic",
        }
    raise ValueError(f"Unknown dataset: {name}")


def ingest_split(name: str, hf_path: str, hf_config, split: str,
                 neo4j: Neo4jManager, parser: DocParser,
                 limit: int = None,
                 shared_vi: VectorIndexManager = None) -> list:
    """
    Ingest one dataset split.
    Returns a list of eval records (with answers) for local JSON storage.
    """
    print(f"\n── {name} / {split} ──")
    try:
        ds = load_dataset(hf_path, hf_config, split=split, cache_dir="data/raw")
    except Exception as e:
        print(f"  Failed to load {name}/{split}: {e}")
        return []

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    processed_docs = set()
    eval_records = []

    for idx, item in enumerate(tqdm(ds, desc=f"{name}/{split}")):
        try:
            norm = norm_item(name, item)
        except Exception as e:
            print(f"  Skipping item {idx}: {e}")
            continue

        doc_id = f"{name}_{norm['raw_doc_id']}"
        image_path = os.path.join(IMAGE_DIR, f"{doc_id}.png")

        # ── Ingest document once ──────────────────────────────────────────
        if doc_id not in processed_docs:
            try:
                img = norm["image"]
                if img is None:
                    continue
                if not os.path.exists(image_path):
                    img.save(image_path)

                neo4j.create_document_node(
                    doc_id=doc_id,
                    filename=f"{doc_id}.png",
                    page_count=1,
                    image_path=image_path,
                )

                doc_data = parser.process_image(image_path)
                text_blocks, block_ids = [], []

                for page in doc_data.get("pages", []):
                    neo4j.create_page_node(doc_id, page["page_no"],
                                           page["width"], page["height"])

                for block in doc_data.get("blocks", []):
                    bbox_hash = hashlib.md5(str(block.get("bbox")).encode()).hexdigest()[:8]
                    block_id = f"{doc_id}_p{block['page']}_b{bbox_hash}"
                    neo4j.create_block_node(doc_id, block["page"], block)
                    if block["type"] == "text":
                        text_blocks.append(block)
                        block_ids.append(block_id)

                if text_blocks:
                    # Re-use the shared model; create fresh index per doc
                    vi = VectorIndexManager() if shared_vi is None else \
                         VectorIndexManager.__new__(VectorIndexManager)
                    if shared_vi is not None:
                        import faiss as _faiss
                        vi.model = shared_vi.model
                        vi.dimension = shared_vi.dimension
                        vi.index = _faiss.IndexFlatL2(shared_vi.dimension)
                        vi.id_map = []
                    vi.add_blocks(text_blocks, block_ids)
                    vi.save(os.path.join(INDEX_DIR, doc_id))

                processed_docs.add(doc_id)
            except Exception as e:
                print(f"  Doc ingestion error for {doc_id}: {e}")
                continue

        # ── Ingest question WITHOUT answer ────────────────────────────────
        q_id = f"{doc_id}_eval_{idx}"
        try:
            neo4j.ingest_eval_question(
                doc_id=doc_id,
                q_id=q_id,
                question_text=norm["question"],
                dataset=name,
                split=split,
            )
        except Exception as e:
            print(f"  Question ingestion error for {q_id}: {e}")

        # ── Store answer locally only ─────────────────────────────────────
        eval_records.append({
            "doc_id":     doc_id,
            "q_id":       q_id,
            "query":      norm["question"],
            "answer":     norm["answers"][0] if norm["answers"] else "",
            "all_answers": norm["answers"],
            "category":   f"{name}/{norm['category']}",
            "image_path": image_path,
            "dataset":    name,
            "split":      split,
        })

    print(f"  Ingested {len(processed_docs)} documents, {len(eval_records)} questions.")
    return eval_records


def main(limit: int = None):
    neo4j = Neo4jManager()
    neo4j.setup_database()
    parser = DocParser(use_ocr=True)
    shared_vi = VectorIndexManager()  # load sentence transformer once

    dataset_configs = [
        ("DocVQA",         "lmms-lab/DocVQA",              "DocVQA",   ["validation", "test"]),
        ("TextVQA",        "lmms-lab/textvqa",              "default",  ["validation"]),
        ("VMCBench",       "suyc21/VMCBench",               None,       ["test"]),
        ("InfographicVQA", "ayoubkirouane/infographic-VQA", None,       ["train"]),
    ]

    test_records, val_records = [], []

    for name, hf_path, hf_config, splits in dataset_configs:
        for split in splits:
            records = ingest_split(name, hf_path, hf_config, split,
                                   neo4j, parser, limit=limit, shared_vi=shared_vi)
            if split in ("test", "train"):
                test_records.extend(records)
            else:
                val_records.extend(records)

    with open("data/processed/test_data_eval.json", "w") as f:
        json.dump(test_records, f, indent=2)
    with open("data/processed/val_data_eval.json", "w") as f:
        json.dump(val_records, f, indent=2)

    print(f"\nDone.")
    print(f"  Test records : {len(test_records)}  → data/processed/test_data_eval.json")
    print(f"  Val  records : {len(val_records)}   → data/processed/val_data_eval.json")
    print(f"  Neo4j graph  : answers NOT stored (questions only)")
    neo4j.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest test/val datasets into Neo4j without correct answers.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max items per split (for quick testing)")
    args = p.parse_args()
    main(args.limit)
