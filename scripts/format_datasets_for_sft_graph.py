import os
import argparse
import hashlib
import uuid
import numpy as np
from PIL import Image
from datasets import load_dataset, concatenate_datasets
from src.graph.neo4j_manager import Neo4jManager
from src.ingestion.parser import DocParser
from src.vision.visualize import MeshVisualizer

def init_managers():
    global neo4j_manager, parser, visualizer
    neo4j_manager = Neo4jManager()
    neo4j_manager.setup_database()
    parser = DocParser(use_ocr=True, use_vision=False)
    visualizer = MeshVisualizer()

def process_with_graph(example, dataset_name):
    # Get image
    img = example['image']
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Generate unique document ID
    doc_id = f"{dataset_name}_{uuid.uuid4().hex[:8]}"
    
    # 1. OCR to extract blocks
    img_np = np.array(img)
    ocr_result = parser.ocr.ocr(img_np, cls=True)
    
    blocks = []
    if ocr_result and ocr_result[0]:
        for idx, line in enumerate(ocr_result[0]):
            bbox, (text, conf) = line
            x0, y0, x1, y1 = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            block = {
                "page": 0,
                "bbox": [x0, y0, x1, y1],
                "type": "text",
                "content": text,
                "confidence": conf
            }
            blocks.append(block)
            
    # 2. Ingest to Neo4j Provenance Graph
    neo4j_manager.create_page_node(doc_id, 0, img.width, img.height)
    for b in blocks:
        neo4j_manager.create_block_node(doc_id, 0, b)
        
    # 3. Retrieve textual graph context
    # Fetch blocks and NEXT relationships
    query = """
    MATCH (p:Page {id: $page_id})-[:CONTAINS]->(b:Block)
    OPTIONAL MATCH (b)-[r:NEXT]->(b2:Block)
    RETURN b.id as id, b.text as text, collect(type(r)) as rels, collect(b2.id) as targets
    """
    graph_data = neo4j_manager.query(query, {"page_id": f"{doc_id}_p0"})
    
    graph_lines = []
    for node in graph_data:
        # Simplify ID for prompt
        short_id = node['id'].split('_')[-1]
        graph_lines.append(f"Node({short_id}, text='{node['text'][:100]}')")
        for rel, target in zip(node['rels'], node['targets']):
            t_short = target.split('_')[-1]
            graph_lines.append(f"Edge({short_id} -[:{rel}]-> {t_short})")
    
    graph_context = "\\n".join(graph_lines)

    # 4. Create 3D Graph representation on Image
    nodes = []
    relationships = []
    for i, b in enumerate(blocks):
        nodes.append({
            "bbox": b["bbox"],
            "label": f"B{i}",
            "type": "evidence"
        })
        if i > 0:
            relationships.append((i-1, i, "NEXT"))
            
    overlay_img = visualizer.draw_mesh_on_image(img.copy(), nodes, relationships, (img.width, img.height))
    
    return doc_id, overlay_img, graph_context

def format_docvqa_graph(example):
    doc_id, overlay_img, graph_context = process_with_graph(example, "DocVQA")
    prompt = f"<|system|>\nYou are a document analysis assistant. Use the following provenance graph to ground your answer.\n[PROVENANCE_GRAPH]\n{graph_context}\n<|user|>\n<|image|>\nQuestion: {example['question']}\n<|assistant|>\n"
    answer = example['answers'][0] if example['answers'] else "N/A"
    sub_category = example['question_types'][0] if example.get('question_types') and len(example['question_types']) > 0 else "General"
    return {"text": prompt + answer, "image": overlay_img, "dataset": "DocVQA", "category": sub_category}

def format_textvqa_graph(example):
    doc_id, overlay_img, graph_context = process_with_graph(example, "TextVQA")
    prompt = f"<|system|>\nYou are a document analysis assistant. Use the following provenance graph to ground your answer.\n[PROVENANCE_GRAPH]\n{graph_context}\n<|user|>\n<|image|>\nQuestion: {example['question']}\n<|assistant|>\n"
    answer = example['answers'][0] if example['answers'] else "N/A"
    sub_category = example['image_classes'][0] if example.get('image_classes') and len(example['image_classes']) > 0 else "General"
    return {"text": prompt + answer, "image": overlay_img, "dataset": "TextVQA", "category": sub_category}

def format_infographicvqa_graph(example):
    # InfographicVQA image might be in a list
    img = example['images'][0] if isinstance(example['images'], list) else example['images']
    # override image temporarily for process_with_graph
    example_copy = example.copy()
    example_copy['image'] = img
    doc_id, overlay_img, graph_context = process_with_graph(example_copy, "InfographicVQA")
    
    user_msg = example['texts'][0]['user']
    assistant_msg = example['texts'][0]['assistant']
    prompt = f"<|system|>\nYou are a document analysis assistant. Use the following provenance graph to ground your answer.\n[PROVENANCE_GRAPH]\n{graph_context}\n<|user|>\n<|image|>\n{user_msg}\n<|assistant|>\n"
    return {"text": prompt + assistant_msg, "image": overlay_img, "dataset": "InfographicVQA", "category": "Infographic"}

def format_vmcbench_graph(example):
    doc_id, overlay_img, graph_context = process_with_graph(example, "VMCBench")
    options = f"A: {example['A']}, B: {example['B']}, C: {example['C']}, D: {example['D']}"
    prompt = f"<|system|>\nYou are a document analysis assistant. Use the following provenance graph to ground your answer.\n[PROVENANCE_GRAPH]\n{graph_context}\n<|user|>\n<|image|>\nQuestion: {example['question']}\nOptions: {options}\nTask: Determine the correct answer.\n<|assistant|>\nCORRECT_OPTION: {example['answer']}"
    return {"text": prompt, "image": overlay_img, "dataset": "VMCBench", "category": example['category']}


def main(limit=None):
    print(f"Loading and formatting datasets with 3D Graph (limit={limit})...")
    init_managers()
    cache_dir = "data/raw"
    processed_dir = "data/processed/multimodal_sft_dataset_graph"
    os.makedirs(processed_dir, exist_ok=True)

    # Note: For processing graph and Neo4j connections properly, num_proc=1 is safer
    # 1. DocVQA
    print("Processing DocVQA...")
    docvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", cache_dir=cache_dir, split="validation")
    if limit: docvqa = docvqa.select(range(min(limit, len(docvqa))))
    docvqa_formatted = docvqa.map(format_docvqa_graph, remove_columns=docvqa.column_names, num_proc=1)

    # 2. TextVQA
    print("Processing TextVQA...")
    textvqa = load_dataset("lmms-lab/textvqa", "default", cache_dir=cache_dir, split="validation")
    if limit: textvqa = textvqa.select(range(min(limit, len(textvqa))))
    textvqa_formatted = textvqa.map(format_textvqa_graph, remove_columns=textvqa.column_names, num_proc=1)

    # 3. InfographicVQA
    print("Processing InfographicVQA...")
    info_vqa = load_dataset("ayoubkirouane/infographic-VQA", cache_dir=cache_dir, split="train")
    if limit: info_vqa = info_vqa.select(range(min(limit, len(info_vqa))))
    info_vqa_formatted = info_vqa.map(format_infographicvqa_graph, remove_columns=info_vqa.column_names, num_proc=1)

    # 4. VMCBench
    print("Processing VMCBench...")
    vmc_bench = load_dataset("suyc21/VMCBench", cache_dir=cache_dir, split="dev")
    if limit: vmc_bench = vmc_bench.select(range(min(limit, len(vmc_bench))))
    vmc_bench_formatted = vmc_bench.map(format_vmcbench_graph, remove_columns=vmc_bench.column_names, num_proc=1)

    # Combine all
    print("Combining datasets...")
    combined_ds = concatenate_datasets([docvqa_formatted, textvqa_formatted, info_vqa_formatted, vmc_bench_formatted])
    
    combined_ds = combined_ds.shuffle(seed=42)

    print(f"Total samples: {len(combined_ds)}")
    combined_ds.save_to_disk(processed_dir)
    print(f"Dataset saved to {processed_dir}")
    
    neo4j_manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per dataset for testing")
    args = parser.parse_args()
    main(args.limit)
