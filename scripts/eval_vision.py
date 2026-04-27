"""
Vision hallucination evaluation: CHAIR, CAOS, NOPE, I-HallA, OOD-refusal, ROUGE/BLEU.
Produces a single JSON report and prints a summary table.
"""
import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import MllamaProcessor, MllamaForConditionalGeneration
from peft import PeftModel
import evaluate
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# COCO-80 object vocabulary used for CHAIR
# ---------------------------------------------------------------------------
COCO_OBJECTS = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
}


def objects_in_text(text: str) -> set:
    words = set(text.lower().split())
    found = set()
    for obj in COCO_OBJECTS:
        if all(w in words for w in obj.split()):
            found.add(obj)
    return found


def chair_score(generated: str, ground_truth_objects: list) -> dict:
    """
    CHAIR_i: fraction of mentioned objects that are hallucinated.
    CHAIR_s: 1 if any hallucination exists, 0 otherwise.
    """
    mentioned = objects_in_text(generated)
    gt_set    = set(g.lower() for g in ground_truth_objects)
    if not mentioned:
        return {"chair_i": 0.0, "chair_s": 0, "hallucinated": [], "mentioned": []}
    hallucinated = mentioned - gt_set
    chair_i = len(hallucinated) / len(mentioned)
    chair_s = int(bool(hallucinated))
    return {
        "chair_i":    chair_i,
        "chair_s":    chair_s,
        "hallucinated": list(hallucinated),
        "mentioned":    list(mentioned),
    }


def caos_score(generated: str, reference: str) -> float:
    """
    Context-Aware Object Similarity: Jaccard similarity of object sets
    between generated and reference text.
    """
    gen_objs = objects_in_text(generated)
    ref_objs = objects_in_text(reference)
    if not gen_objs and not ref_objs:
        return 1.0
    if not gen_objs or not ref_objs:
        return 0.0
    return len(gen_objs & ref_objs) / len(gen_objs | ref_objs)


def is_ood_refusal(text: str) -> bool:
    """Return True if the model correctly refuses an out-of-domain query."""
    t = text.lower()
    refusal_phrases = [
        "no mention", "not mentioned", "not in the document", "cannot find",
        "not present", "document does not", "no information", "not available",
        "not found in", "outside the scope",
    ]
    return any(p in t for p in refusal_phrases)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_id: str, adapter_path: str = None):
    hf_token = os.getenv("HF_TOKEN")
    processor = MllamaProcessor.from_pretrained(model_id, token=hf_token)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", token=hf_token
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model.eval(), processor


def run_inference(model, processor, messages: list, image: Image.Image = None,
                  max_new_tokens: int = 80) -> str:
    device = next(model.parameters()).device
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    if image is not None:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(text=prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
    
    generated = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in generated:
        return generated.split("assistant")[-1].strip()
    else:
        full_decoded = processor.decode(output[0], skip_special_tokens=False)
        if "<|start_header_id|>assistant<|end_header_id|>\n\n" in full_decoded:
            return full_decoded.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
        else:
            input_decoded = processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
            return generated[len(input_decoded):].strip()


# ---------------------------------------------------------------------------
# Evaluation runners
# ---------------------------------------------------------------------------

def eval_chair_caos(model, processor, test_items: list) -> list:
    """
    Items must have: image_path, caption_prompt, ground_truth_objects (list of strings).
    """
    results = []
    for item in tqdm(test_items, desc="CHAIR/CAOS"):
        image = Image.open(item["image_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Describe what you see in this image."}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item.get('caption_prompt', 'Describe the image.')}
            ]}
        ]
        generated = run_inference(model, processor, messages, image, max_new_tokens=100)
        gt_objects = item.get("ground_truth_objects", [])
        reference  = item.get("reference_caption", " ".join(gt_objects))
        c = chair_score(generated, gt_objects)
        results.append({
            "type":      "CHAIR_CAOS",
            "generated": generated,
            "chair_i":   c["chair_i"],
            "chair_s":   c["chair_s"],
            "caos":      caos_score(generated, reference),
            "hallucinated_objects": c["hallucinated"],
        })
    return results


def eval_nope(model, processor, test_items: list) -> list:
    """
    Items must have: image_path, absent_object (the object NOT in the image), question.
    """
    results = []
    for item in tqdm(test_items, desc="NOPE"):
        image = Image.open(item["image_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Answer precisely. If something is not in the image, say so."}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item['question']}
            ]}
        ]
        generated = run_inference(model, processor, messages, image)
        absent = item["absent_object"].lower()
        # Hallucination = model mentions the absent object affirmatively
        hallucinated = absent in generated.lower() and "no" not in generated.lower()[:30]
        results.append({
            "type":         "NOPE",
            "absent_object": item["absent_object"],
            "generated":    generated,
            "hallucinated": hallucinated,
            "is_correct":   not hallucinated,
        })
    return results


def eval_ihalla(model, processor, test_items: list) -> list:
    """
    I-HallA: tricky VQA items where the correct answer is that the object/detail is absent.
    Items: image_path, question, expected_answer, is_negative (bool).
    """
    rouge = evaluate.load("rouge")
    bleu  = evaluate.load("bleu")
    results = []
    for item in tqdm(test_items, desc="I-HallA"):
        image = Image.open(item["image_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Answer the question about the image factually."}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": item['question']}
            ]}
        ]
        generated = run_inference(model, processor, messages, image)
        expected  = item.get("expected_answer", "")
        is_correct = expected.lower() in generated.lower() if expected else False

        rouge_s = rouge.compute(predictions=[generated], references=[expected]).get("rougeL", 0)
        bleu_s  = bleu.compute(predictions=[generated], references=[[expected]]).get("bleu", 0)

        results.append({
            "type":       "I-HallA",
            "question":   item["question"],
            "expected":   expected,
            "generated":  generated,
            "is_correct": is_correct,
            "rougeL":     rouge_s,
            "bleu":       bleu_s,
        })
    return results


def eval_ood_refusal(model, processor, test_items: list) -> list:
    """
    OOD hallucination: model must refuse queries about content not in the document
    and give a reasoning grounded in the document context.
    Items: image_path (optional), question (out-of-domain), doc_context.
    """
    results = []
    for item in tqdm(test_items, desc="OOD-Refusal"):
        doc_context = item.get("doc_context", "")
        image_path  = item.get("image_path")
        image = Image.open(image_path).convert("RGB") if image_path else None

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a document analysis assistant. Only answer questions based on the provided document context. If the document does not contain information about the question, respond: 'Based on the document, there is no mention of [topic]. The document discusses [brief summary].'"}]},
        ]
        user_content = []
        if image is not None:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Document context: {doc_context}\nQuestion: {item['question']}"})
        messages.append({"role": "user", "content": user_content})

        generated = run_inference(model, processor, messages, image, max_new_tokens=120)
        refused  = is_ood_refusal(generated)
        has_reason = any(w in generated.lower() for w in ["because", "however", "instead", "discusses", "document"])

        results.append({
            "type":        "OOD_Refusal",
            "question":    item["question"],
            "generated":   generated,
            "refused":     refused,
            "has_reason":  has_reason,
            "is_correct":  refused,
        })
    return results


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def aggregate(items: list, field: str = "is_correct") -> float:
    if not items:
        return 0.0
    return float(np.mean([r.get(field, 0) for r in items]))


def build_test_data_from_docvqa(image_dir: str, limit: int = 20):
    """
    Build synthetic test items from docvqa images when no dedicated test set exists.
    """
    items = {"chair_caos": [], "nope": [], "ihalla": [], "ood": []}
    images = [f for f in os.listdir(image_dir) if f.endswith(".png")][:limit]

    absent_objects_pool = ["giraffe", "airplane", "snowboard", "toothbrush", "skateboard"]

    for idx, fname in enumerate(images):
        img_path = os.path.join(image_dir, fname)
        items["chair_caos"].append({
            "image_path":         img_path,
            "caption_prompt":     "Describe the image.",
            "ground_truth_objects": [],   # populated from annotation if available
            "reference_caption":  "",
        })
        items["nope"].append({
            "image_path":    img_path,
            "absent_object": absent_objects_pool[idx % len(absent_objects_pool)],
            "question":      f"Is there a {absent_objects_pool[idx % len(absent_objects_pool)]} in this image?",
        })
        items["ihalla"].append({
            "image_path":      img_path,
            "question":        "What color is the sky in this image?",
            "expected_answer": "no sky",
            "is_negative":     True,
        })
        items["ood"].append({
            "image_path": img_path,
            "question":   "What is the GDP of France mentioned in this document?",
            "doc_context": "This is a document image.",
        })
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",     default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_data",    default=None,  help="JSON with test items")
    parser.add_argument("--image_dir",    default="data/processed/docvqa_images")
    parser.add_argument("--output_path",  default="logs/eval_vision_results.json")
    parser.add_argument("--limit",        type=int, default=20)
    args = parser.parse_args()

    model, processor = load_model(args.model_id, args.adapter_path)

    if args.test_data and os.path.exists(args.test_data):
        with open(args.test_data) as f:
            test_items = json.load(f)
    else:
        print(f"No test_data provided; generating synthetic items from {args.image_dir}")
        test_items = build_test_data_from_docvqa(args.image_dir, limit=args.limit)

    # Run evaluations
    chair_caos_results = eval_chair_caos(model, processor, test_items.get("chair_caos", []))
    nope_results       = eval_nope(model, processor, test_items.get("nope", []))
    ihalla_results     = eval_ihalla(model, processor, test_items.get("ihalla", []))
    ood_results        = eval_ood_refusal(model, processor, test_items.get("ood", []))

    report = {
        "model_id":     args.model_id,
        "adapter_path": args.adapter_path,
        "CHAIR": {
            "chair_i_mean": float(np.mean([r["chair_i"] for r in chair_caos_results])) if chair_caos_results else 0.0,
            "chair_s_mean": float(np.mean([r["chair_s"] for r in chair_caos_results])) if chair_caos_results else 0.0,
        },
        "CAOS": {
            "caos_mean": float(np.mean([r["caos"] for r in chair_caos_results])) if chair_caos_results else 0.0,
        },
        "NOPE": {
            "accuracy":          aggregate(nope_results),
            "hallucination_rate": 1.0 - aggregate(nope_results),
            "total":             len(nope_results),
        },
        "I_HallA": {
            "accuracy": aggregate(ihalla_results),
            "rougeL":   float(np.mean([r["rougeL"] for r in ihalla_results])) if ihalla_results else 0.0,
            "bleu":     float(np.mean([r["bleu"]   for r in ihalla_results])) if ihalla_results else 0.0,
            "total":    len(ihalla_results),
        },
        "OOD_Refusal": {
            "refusal_rate":    aggregate(ood_results, "refused"),
            "reasoning_rate":  aggregate(ood_results, "has_reason"),
            "total":           len(ood_results),
        },
        "details": {
            "chair_caos": chair_caos_results,
            "nope":       nope_results,
            "ihalla":     ihalla_results,
            "ood":        ood_results,
        },
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nVision Evaluation → {args.output_path}")
    print(f"  CHAIR-i (lower=better): {report['CHAIR']['chair_i_mean']:.4f}")
    print(f"  CHAIR-s (lower=better): {report['CHAIR']['chair_s_mean']:.4f}")
    print(f"  CAOS    (higher=better): {report['CAOS']['caos_mean']:.4f}")
    print(f"  NOPE accuracy:           {report['NOPE']['accuracy']:.4f}")
    print(f"  I-HallA accuracy:        {report['I_HallA']['accuracy']:.4f}")
    print(f"  OOD refusal rate:        {report['OOD_Refusal']['refusal_rate']:.4f}")
    print(f"  OOD reasoning rate:      {report['OOD_Refusal']['reasoning_rate']:.4f}")


if __name__ == "__main__":
    main()
