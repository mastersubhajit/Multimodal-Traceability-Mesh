"""
Comprehensive evaluation metrics for Multimodal Traceability Mesh.
Covers: retrieval, MCQ accuracy (CSQA/OBQA style), text generation, FActScore,
citation accuracy, graph comprehension, hallucination (CHAIR/CAOS/NOPE/I-HallA),
OOD refusal, RAGAS, and latency tracking.
"""
import re
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter


# ── Retrieval ────────────────────────────────────────────────────────────────

def calculate_recall(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    if not ground_truth_ids:
        return 0.0
    return len(set(retrieved_ids) & set(ground_truth_ids)) / len(ground_truth_ids)


def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    gt = set(ground_truth_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in gt:
            return 1.0 / (i + 1)
    return 0.0


def calculate_hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = None) -> float:
    subset = retrieved_ids[:k] if k else retrieved_ids
    return 1.0 if any(r in set(ground_truth_ids) for r in subset) else 0.0


# ── Exact match / accuracy ────────────────────────────────────────────────────

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    if not predictions or len(predictions) != len(ground_truths):
        return 0.0
    return float(np.mean([calculate_exact_match(p, g) for p, g in zip(predictions, ground_truths)]))


# ── MCQ option extraction (CSQA / OBQA / VMCBench style) ────────────────────

def _extract_option(text: str) -> str:
    """Extract the first A–E answer letter from model output."""
    t = text.strip().upper()
    for pat in [
        r"CORRECT[_\s]OPTION\s*[:\s]+([A-E])",
        r"CORRECT\s+ANSWER\s*[:\s]+([A-E])",
        r"ANSWER\s*[:\s]+([A-E])",
        r"^([A-E])\s*[\.:\)]",
        r"\bOPTION\s+([A-E])\b",
        r"\b([A-E])\b",
    ]:
        m = re.search(pat, t)
        if m:
            return m.group(1)
    return ""


def calculate_mcq_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Accuracy for MCQ tasks — works for CSQA, OBQA, VMCBench."""
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, g in zip(predictions, ground_truths)
        if _extract_option(p) == g.strip().upper()[:1]
    )
    return correct / len(predictions)


def calculate_mrr_mcq(predictions: List[str], ground_truths: List[str]) -> float:
    """MRR for MCQ: reciprocal rank of the correct option in the order options are mentioned."""
    rr_sum = 0.0
    for pred, gt in zip(predictions, ground_truths):
        t = pred.strip().upper()
        gt_ch = gt.strip().upper()[:1]
        seen = sorted((t.find(ch), ch) for ch in "ABCDE" if t.find(ch) != -1)
        labels = [ch for _, ch in seen]
        try:
            rr_sum += 1.0 / (labels.index(gt_ch) + 1)
        except ValueError:
            pass
    return rr_sum / len(predictions) if predictions else 0.0


# ── ROUGE / BLEU ──────────────────────────────────────────────────────────────

def calculate_rouge_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-L and BLEU using the `evaluate` library."""
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        r = rouge.compute(predictions=predictions, references=references)
        b = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
        return {"rougeL": float(r.get("rougeL", 0.0)), "bleu": float(b.get("bleu", 0.0))}
    except Exception as e:
        # Fallback with rouge_score library
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
            return {"rougeL": float(np.mean([s["rougeL"].fmeasure for s in scores])), "bleu": 0.0}
        except Exception:
            return {"rougeL": 0.0, "bleu": 0.0, "error": str(e)}


# ── FActScore ─────────────────────────────────────────────────────────────────

def calculate_factscore(
    predictions: List[str],
    evidences: List[str],
    sentence_threshold: float = 0.3,
) -> float:
    """
    Approximate FActScore: fraction of generated sentences whose token-level
    recall against the provided evidence exceeds `sentence_threshold`.
    No external API required.
    """
    import nltk
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            break
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)

    scores = []
    for pred, ev in zip(predictions, evidences):
        if not pred or not ev:
            scores.append(0.0)
            continue
        try:
            sents = nltk.sent_tokenize(pred)
        except Exception:
            sents = [pred]
        ev_tokens = set(ev.lower().split())
        if not ev_tokens:
            scores.append(0.0)
            continue
        supported = 0
        for sent in sents:
            sent_tokens = set(sent.lower().split())
            if sent_tokens and len(sent_tokens & ev_tokens) / len(sent_tokens) >= sentence_threshold:
                supported += 1
        scores.append(supported / len(sents) if sents else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ── Citation accuracy ─────────────────────────────────────────────────────────

def calculate_citation_accuracy(generated_texts: List[str], evidence_lists: List[List[str]]) -> float:
    """
    Fraction of [n] citation markers in the generated text that map to a
    valid (in-range) evidence index.
    """
    scores = []
    for gen, evs in zip(generated_texts, evidence_lists):
        cited = [int(m) - 1 for m in re.findall(r'\[(\d+)\]', gen) if m.isdigit()]
        if not cited:
            scores.append(0.0)
            continue
        valid = sum(1 for idx in cited if 0 <= idx < len(evs))
        scores.append(valid / len(cited))
    return float(np.mean(scores)) if scores else 0.0


# ── Graph comprehension ───────────────────────────────────────────────────────

GRAPH_TASK_TYPES = [
    "N. description",
    "N. degree",
    "Highest N. degree",
    "N. number",
    "E. number",
    "Triple listing",
]


def calculate_graph_type_accuracy(results: List[Dict]) -> Dict[str, float]:
    """
    Per-task-type accuracy for provenance graph comprehension.
    Each result dict must have: type (str), is_correct (bool).
    """
    stats: Dict[str, Dict] = {}
    for r in results:
        t = r.get("type", "unknown")
        if t not in stats:
            stats[t] = {"correct": 0, "total": 0}
        stats[t]["total"] += 1
        if r.get("is_correct", False):
            stats[t]["correct"] += 1
    return {t: v["correct"] / v["total"] if v["total"] else 0.0 for t, v in stats.items()}


# ── CHAIR ─────────────────────────────────────────────────────────────────────

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


def _objects_in_text(text: str) -> set:
    words = set(text.lower().split())
    return {obj for obj in COCO_OBJECTS if all(w in words for w in obj.split())}


def chair_score(generated: str, ground_truth_objects: List[str]) -> Dict:
    """
    CHAIR-i: fraction of mentioned COCO objects that are hallucinated.
    CHAIR-s: 1 if any hallucination exists in the caption.
    """
    mentioned = _objects_in_text(generated)
    gt_set = {g.lower() for g in ground_truth_objects}
    if not mentioned:
        return {"chair_i": 0.0, "chair_s": 0, "hallucinated": [], "mentioned": []}
    hallucinated = mentioned - gt_set
    return {
        "chair_i": len(hallucinated) / len(mentioned),
        "chair_s": int(bool(hallucinated)),
        "hallucinated": list(hallucinated),
        "mentioned": list(mentioned),
    }


# ── CAOS ──────────────────────────────────────────────────────────────────────

def caos_score(generated: str, reference: str) -> float:
    """Context-Aware Object Similarity: Jaccard similarity of COCO object sets."""
    gen_objs = _objects_in_text(generated)
    ref_objs = _objects_in_text(reference)
    if not gen_objs and not ref_objs:
        return 1.0
    if not gen_objs or not ref_objs:
        return 0.0
    return len(gen_objs & ref_objs) / len(gen_objs | ref_objs)


# ── NOPE ──────────────────────────────────────────────────────────────────────

_NEGATION_WORDS = {"no ", "not ", "don't", "cannot", "absent", "none ", "never ", "n't "}


def nope_score(generated: str, absent_object: str) -> Dict:
    """
    Negative Object Presence Evaluation.
    is_correct=True means the model correctly did NOT hallucinate the absent object.
    """
    absent = absent_object.lower()
    gen_lower = generated.lower()
    pos = gen_lower.find(absent)
    if pos == -1:
        hallucinated = False
    else:
        window = gen_lower[max(0, pos - 30): pos + len(absent) + 10]
        hallucinated = not any(n in window for n in _NEGATION_WORDS)
    return {
        "hallucinated": hallucinated,
        "is_correct": not hallucinated,
        "score": 0.0 if hallucinated else 1.0,
    }


# ── I-HallA ───────────────────────────────────────────────────────────────────

def ihalla_score(generated: str, expected: str) -> Dict:
    """
    Image Hallucination with VQA.
    The model must acknowledge an absent visual detail (expected is the correct answer).
    """
    is_correct = expected.strip().lower() in generated.lower()
    return {"is_correct": is_correct, "score": 1.0 if is_correct else 0.0}


# ── OOD refusal ───────────────────────────────────────────────────────────────

_OOD_PHRASES = [
    "no mention", "not mentioned", "not in the document", "document does not",
    "not available", "not found", "cannot find", "no information",
    "outside the scope", "not present", "document has no", "in the document it has no",
    "the document does not contain", "no record",
]

_REASONING_WORDS = [
    "because", "however", "instead", "document discusses", "document covers",
    "the document", "based on", "since", "as the document", "the provided",
]


def ood_refusal_score(generated: str) -> Dict:
    """
    Check if the model correctly refuses an OOD query and provides reasoning.
    score: 0 = not refused, 1 = refused, 1.5 = refused + reasoning present.
    """
    t = generated.lower()
    refused = any(p in t for p in _OOD_PHRASES)
    has_reasoning = refused and any(w in t for w in _REASONING_WORDS)
    return {
        "refused": refused,
        "has_reasoning": has_reasoning,
        "is_correct": refused,
        "score": (1.0 if refused else 0.0) + (0.5 if has_reasoning else 0.0),
    }


# ── RAGAS wrapper ─────────────────────────────────────────────────────────────

def compute_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[Any],
    ground_truths: List[str],
) -> Dict[str, float]:
    """Compute RAGAS faithfulness/answer_relevancy/context_precision/context_recall."""
    try:
        from ragas import evaluate as ragas_eval
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset as HFDataset
        ds = HFDataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": [[c] if isinstance(c, str) else list(c) for c in contexts],
            "ground_truth": ground_truths,
        })
        scores = ragas_eval(ds, metrics=[faithfulness, answer_relevancy,
                                         context_precision, context_recall])
        return {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
    except Exception as e:
        return {"ragas_skipped": str(e)}


# ── Aggregate helper ──────────────────────────────────────────────────────────

def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Compute the mean of every numeric field across a list of per-sample dicts."""
    if not results:
        return {}
    numeric_keys = [k for k in results[0] if isinstance(results[0].get(k), (int, float))]
    return {k: float(np.mean([r.get(k, 0.0) for r in results])) for k in numeric_keys}
