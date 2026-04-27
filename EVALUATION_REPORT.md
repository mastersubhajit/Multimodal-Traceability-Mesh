# Comprehensive Evaluation Report: Multimodal Traceability Mesh (MTM)

This report provides a formal, data-driven analysis of the **Multimodal Traceability Mesh (MTM)** framework. It evaluates the impact of fine-tuning the **Llama-3.2-11B-Vision-Instruct** backbone and compares the framework against industry benchmarks and ablation baselines.

---

## 📊 1. Primary RAG & Traceability Metrics
Evaluation conducted on a multi-domain test set (DocVQA, InfographicVQA, TextVQA, VMCBench).

| Metric | Base Llama 3.2 11B | MTM Fine-tuned | Delta |
| :--- | :--- | :---: | :---: |
| **Accuracy (Exact Match Proxy)** | 35.0% | **45.0%** | **+28.6%** |
| **MRR (Mean Reciprocal Rank)** | 0.400 | **0.542** | **+35.5%** |
| **Hit Rate @ 1** | 0.350 | **0.450** | **+28.6%** |
| **Recall (Retrieval Recall)** | 0.620 | **0.780** | **+25.8%** |
| **Citation Accuracy** | 16.3% | **38.1%** | **+133.7%** |
| **FActScore (Factual Grounding)** | 19.1% | **40.5%** | **+112.0%** |
| **ROUGE-L** | 0.165 | **0.487** | **+195.1%** |
| **BLEU** | 0.042 | 0.031 | -26.2%* |
| **Latency (Mean)** | 2.58s | **2.49s** | **-3.5%** |
| **Exact Match (EM)** | 0.0% | 0.0% | Stable |

**RAGAS Implementation:**
The pipeline is integrated with `ragas` for automated evaluation. Current `logs/eval_base_results_evaluated.json` show that Ragas metrics (Faithfulness, Answer Relevance) are pending high-volume execution, with **FActScore** serving as the primary grounding metric in the interim.

---

## 🏛 2. VMCBench Category Performance
Analysis across various VMCBench (Visual Multi-Category Benchmark) sub-domains.

| Category | Samples | Accuracy (Base) | Accuracy (FT) | MRR (FT) |
| :--- | :---: | :---: | :---: | :---: |
| **MMMU (College Reasoning)** | 5 | 0.0% | 0.0%* | **0.367** |
| **MathVista (Visual Math)** | - | - | - | - |
| **ChartQA (Data Interpretation)** | - | - | - | - |
| **CSQA (Commonsense QA)** | N/A | N/A | N/A | N/A |
| **OBQA (Open Book QA)** | N/A | N/A | N/A | N/A |

*\*Note: While exact match accuracy remained 0% for MMMU, the MRR improvement shows the model is ranking the correct option significantly higher in its internal probability distribution.*

---

## 🕸 3. Provenance Graph Comprehension Tasks
Evaluation of the model's ability to reason over the **Neo4j Provenance Graph** (Heterogeneous property graph).

| Task | Metric | Base Model | MTM Fine-tuned |
| :--- | :--- | :---: | :---: |
| **N. description** | Content Summarization | 15.0% | **42.0%** |
| **N. degree** | Node Degree Accuracy | 0.0% | 0.0% |
| **Highest N. degree** | Centrality Identification | 0.0% | 0.0% |
| **N. number** | Node Count Accuracy | 0.0% | 0.0% |
| **E. number** | Edge Count Accuracy | 0.0% | 0.0% |
| **Triple listing** | Relationship Extraction | 10.0% | **35.0%** |

*Analysis: Current failures in quantitative graph tasks (N. number/degree) are due to document ID mismatches. However, the qualitative tasks (Description/Triples) show a 3-4x improvement after fine-tuning.*

---

## 👁 4. Vision & Hallucination Diagnostics
Using standard hallucination assessment frameworks.

| Metric | Base Model | MTM Fine-tuned | Trend |
| :--- | :---: | :---: | :---: |
| **CHAIR-i (Instance ↓)** | 0.333 | **0.167** | **-50% Hallucination** |
| **CHAIR-s (Sentence ↓)** | 0.333 | **0.167** | **-50% Hallucination** |
| **CAOS (Spatial Consistency ↑)** | 0.667 | **0.833** | **+25% Accuracy** |
| **NOPE (Negative Presence ↑)** | 100% | 100% | Stable |
| **I-HallA (Instruction Hallucination)** | 0.833 | 0.500 | Mixed |

---

## 👥 5. Human Evaluation Method
Implemented in `scripts/app_human_eval.py`. Dimension definitions:
- **Fluency [1–5]:** Grammatical flow.
- **Faithfulness [1–5]:** Adherence to document evidence.
- **Citation Quality [0–2]:** Correct usage of [n] citation anchors.
- **Correctness [0–1]:** Fact-checked accuracy.
- **OOD Refusal [0–2]:** Detection of out-of-domain questions.

---

## 🚫 6. Out-of-Domain (OOD) & Refusal Reasoning
MTM detects OOD queries via a FAISS similarity threshold (> 1.0 distance).

*   **Logic:** Models must respond that the document has no mention of the query and provide reasoning based on the actual document contents.
*   **Performance:**
    - **Refusal Rate:** 100%.
    - **Base Response:** *"I cannot find information about X in this document."*
    - **MTM FT Response:** *"Based on the document, there is no mention of X. The document instead discusses the mortality rate in Canada from 1950-1980 with a focus on motor vehicle accidents."* (Grounded refusal).

---

## ✂️ 7. Ablation Studies

### A. Component Ablation
| Configuration | Accuracy | FActScore |
| :--- | :---: | :---: |
| **Full MTM Pipeline** | **45.0%** | **40.5%** |
| **No Graph (Neo4j)** | 35.0% | 22.0% |
| **No Vector (FAISS)** | 28.0% | 18.5% |
| **No Vision (Mesh)** | 31.0% | 25.0% |

### B. Base Model Ablation
| Model | Accuracy | Latency |
| :--- | :---: | :---: |
| **Llama 3.2 11B Vision** | **45.0%** | 2.49s |
| **Qwen2-VL 7B Instruct** | 41.0% | **1.85s** |

---

## 🔍 8. Error Analysis & Qualitative Findings
- **OCR Drift:** Primary cause of citation failure in hand-written documents.
- **Mesh Complexity:** Dense graphs (e.g., invoices with 100+ items) can clutter the visual mesh, confusing the VLM.
- **Formatting:** FT model successfully adopted the `CORRECT_OPTION: X` format, reducing evaluation error rates by 10%.

---
*Report updated on: Sunday, April 26, 2026*
