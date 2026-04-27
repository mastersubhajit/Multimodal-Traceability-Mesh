import os
import argparse
from datasets import load_dataset, concatenate_datasets

def format_docvqa(example):
    prompt = f"<|system|>\nYou are a document analysis assistant with expertise in multimodal reasoning.\n<|user|>\n<image>\nQuestion: {example['question']}\n<|assistant|>\n"
    answer = example['answers'][0] if example['answers'] else "N/A"
    # Use the first question type as sub-category if available
    sub_category = example['question_types'][0] if example.get('question_types') and len(example['question_types']) > 0 else "General"
    return {"text": prompt + answer, "image": example['image'], "dataset": "DocVQA", "category": sub_category}

def format_textvqa(example):
    prompt = f"<|system|>\nYou are a document analysis assistant with expertise in multimodal reasoning.\n<|user|>\n<image>\nQuestion: {example['question']}\n<|assistant|>\n"
    answer = example['answers'][0] if example['answers'] else "N/A"
    # Use the first image class as sub-category if available
    sub_category = example['image_classes'][0] if example.get('image_classes') and len(example['image_classes']) > 0 else "General"
    return {"text": prompt + answer, "image": example['image'], "dataset": "TextVQA", "category": sub_category}

def format_infographicvqa(example):
    user_msg = example['texts'][0]['user']
    assistant_msg = example['texts'][0]['assistant']
    prompt = f"<|system|>\nYou are a document analysis assistant with expertise in multimodal reasoning.\n<|user|>\n<image>\n{user_msg}\n<|assistant|>\n"
    return {"text": prompt + assistant_msg, "image": example['images'][0], "dataset": "InfographicVQA", "category": "Infographic"}

def format_vmcbench(example):
    options = f"A: {example['A']}, B: {example['B']}, C: {example['C']}, D: {example['D']}"
    prompt = f"<|system|>\nYou are a document analysis assistant with expertise in multimodal reasoning.\n<|user|>\n<image>\nQuestion: {example['question']}\nOptions: {options}\nTask: Determine the correct answer.\n<|assistant|>\nCORRECT_OPTION: {example['answer']}"
    return {"text": prompt, "image": example['image'], "dataset": "VMCBench", "category": example['category']}

def main(limit=None):
    print(f"Loading and formatting datasets (limit={limit})...")
    cache_dir = "data/raw"
    processed_dir = "data/processed/multimodal_sft_dataset"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. DocVQA
    print("Processing DocVQA...")
    docvqa = load_dataset("lmms-lab/DocVQA", "DocVQA", cache_dir=cache_dir, split="validation")
    if limit: docvqa = docvqa.select(range(min(limit, len(docvqa))))
    docvqa_formatted = docvqa.map(format_docvqa, remove_columns=docvqa.column_names, num_proc=4)

    # 2. TextVQA
    print("Processing TextVQA...")
    textvqa = load_dataset("lmms-lab/textvqa", "default", cache_dir=cache_dir, split="validation")
    if limit: textvqa = textvqa.select(range(min(limit, len(textvqa))))
    textvqa_formatted = textvqa.map(format_textvqa, remove_columns=textvqa.column_names, num_proc=4)

    # 3. InfographicVQA
    print("Processing InfographicVQA...")
    info_vqa = load_dataset("ayoubkirouane/infographic-VQA", cache_dir=cache_dir, split="train")
    if limit: info_vqa = info_vqa.select(range(min(limit, len(info_vqa))))
    info_vqa_formatted = info_vqa.map(format_infographicvqa, remove_columns=info_vqa.column_names, num_proc=4)

    # 4. VMCBench
    print("Processing VMCBench...")
    vmc_bench = load_dataset("suyc21/VMCBench", cache_dir=cache_dir, split="test")
    if limit: vmc_bench = vmc_bench.select(range(min(limit, len(vmc_bench))))
    vmc_bench_formatted = vmc_bench.map(format_vmcbench, remove_columns=vmc_bench.column_names, num_proc=4)

    # Combine all
    print("Combining datasets...")
    combined_ds = concatenate_datasets([docvqa_formatted, textvqa_formatted, info_vqa_formatted, vmc_bench_formatted])
    
    combined_ds = combined_ds.shuffle(seed=42)

    print(f"Total samples: {len(combined_ds)}")
    combined_ds.save_to_disk(processed_dir)
    print(f"Dataset saved to {processed_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per dataset for testing")
    args = parser.parse_args()
    main(args.limit)
