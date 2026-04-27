# PROJECT-CONFIG.md — Universal Research Project Configuration
# Fill every section before using /write-chapter.
# Agents read this file before every action.
# Replaces: RQ-SKELETON.md + WORKSPACE-MAPPER.md (v1.0)

---

## 1. Project Identity

Working Title: Multimodal Traceability Mesh (MTM): Structural Grounding for Verifiable Document Analysis
Domain: computer-vision
Target Venue: CVPR / ICCV / NeurIPS
Target Quartile: Q1
Citation Style: IEEE
Document Type: conference-paper

---

## 2. Domain Declaration

domain: computer-vision

This field triggers auto-loading of domain modules from `.agent/domains/[domain]/`.
Use exact folder name. Available: computer-vision, nlp, reinforcement-learning, biomedical, social-science.

---

## 3. Research Problem (50-80 words)

Current Vision-Language Models (VLMs) like Llama 3.2 Vision often struggle with complex document analysis, frequently suffering from object-level hallucinations and failing to grasp structural relationships (Fabbri et al., 2024). When documents are treated as raw pixels or flattened text, the model loses critical layout information and logical provenance (Lee et al., 2024), resulting in answers that are unverified and often spatially inconsistent. This limits the reliability of VLMs in high-stakes document processing tasks where traceability is paramount.

---

## 4. Research Gap

Gap Type: Methodological / Knowledge

While recent VLMs like Llama 3.2 demonstrate impressive multimodal capabilities, they lack a mechanism for structural grounding and verifiable evidence citation. Existing RAG frameworks typically focus on semantic vector similarity, which hits a "single-vector ceiling" for complex document queries (DeepMind, 2025), ignoring the rich structural "DNA" of documents. Furthermore, industry reports indicate a ~26% error rate in LLM citations (CiteFix, 2025), creating a critical gap in verifiable document interpretation that traditional vector-only approaches (ColPali, 2024) have yet to bridge in multi-page, high-stakes contexts.

---

## 5. Research Aim

To develop and evaluate the Multimodal Traceability Mesh (MTM) framework that integrates a Neo4j provenance graph with a fine-tuned Llama 3.2 Vision model to provide grounded, verifiable, and citation-accurate document analysis.

---

## 6. Research Questions

RQ1: To what extent does the integration of a provenance graph reduce object-level hallucinations (CHAIR metrics) in document-based VQA?
RQ2: Does a hybrid Graph-Vector retrieval approach provide more relevant context than standard vector-only retrieval for complex documents?
RQ3: How effectively does a "Visual Mesh" overlay enhance the spatial reasoning and citation accuracy of fine-tuned VLMs?

---

## 7. Hypotheses

H1: Integrating a provenance graph will reduce instance hallucinations (CHAIR-i) by at least 30% compared to a base VLM.
H2: Hybrid retrieval will improve Mean Reciprocal Rank (MRR) by >20% compared to baseline vector search.
H3: The Visual Mesh overlay combined with fine-tuning will increase citation accuracy by over 100%.

---

## 8. RQ-to-Method Mapping

| RQ  | Hypothesis | Method/Experiment | Primary Metric | Chapter |
|-----|-----------|-------------------|---------------|---------|
| RQ1 | H1        | **Hallucination Diagnostic:** CHAIR-i/CAOS evaluation comparing MTM Fine-tuned vs. Base VLM using object-extraction vs. pixel-grounding. | CHAIR-i (Instance) | Ch.4 |
| RQ2 | H2        | **Hybrid Retrieval:** Comparison of Vector-only (FAISS) vs. Hybrid (FAISS + Neo4j 2-hop neighbor traversal) across multi-domain docs. | MRR / Recall@K | Ch.4 |
| RQ3 | H3        | **Spatial Grounding:** Comparative analysis of "Visual Mesh" synthetic image overlays vs. textual-only provenance markers for citation accuracy. | Citation Accuracy | Ch.4 |

---

## 9. Scope

In scope: Multi-domain document VQA (DocVQA, InfographicVQA, TextVQA, VMCBench), Neo4j graph construction, QLoRA fine-tuning of Llama 3.2 11B Vision, Hallucination diagnostics.
Out of scope: Real-time video analysis, non-document multimodal tasks (e.g., natural scene image captioning outside of documents), training from scratch.

---

## 10. Expected Contributions

1. A novel Multimodal Traceability Mesh (MTM) architecture bridging Neo4j graphs and Vision-Language Models.
2. A "Visual Mesh" synthetic overlay technique for spatial grounding of VLMs.
3. Empirical evidence showing significant hallucination reduction and improved citation accuracy through provenance-based fine-tuning.

---

## 11. Dataset / Data

Total samples:     ~10,000 (Aggregated from multiple sources)
Train:             8,000 (80%)
Validation:        1,000 (10%)
Test:              1,000 (10%)
Classes/Categories:DocVQA, InfographicVQA, TextVQA, VMCBench (MathVista, MMMU, ChartQA)
Data format:       Images (Documents) + Structured Metadata (Neo4j Graph)
Source:            Public Benchmarks (lmms-lab, ayoubkirouane, suyc21)

---

## 12. Experiment Configuration

Model/Method:      Llama-3.2-11B-Vision-Instruct (Fine-tuned via QLoRA)
Initialization:    Meta AI Llama 3.2 Vision Checkpoint
Hardware:          ASL-gpu (2x NVIDIA GPUs, 128G RAM)
Key Hyperparameters:
  - learning_rate: 2e-4 (Standard QLoRA)
  - lora_r: 16
  - lora_alpha: 32
  - max_seq_length: 2048
Software:          PyTorch 2.x, Transformers, Neo4j, FAISS, PaddleOCR

---

## 13. Primary Results

File: logs/eval_base_results_evaluated.json

| Metric          | Value   | Conditions                    |
|-----------------|---------|-------------------------------|
| Accuracy        | 45.0%   | MTM Fine-tuned                |
| MRR             | 0.542   | MTM Fine-tuned                |
| Citation Acc    | 38.1%   | MTM Fine-tuned                |
| FActScore       | 40.5%   | MTM Fine-tuned                |
| CHAIR-i         | 0.167   | Hallucination (Lower is better)|

---

## 14. Ablation / Validation Results

File: EVALUATION_REPORT.md

| Configuration        | Accuracy | FActScore | Citation Acc |
|---------------------|------------|------------|------------|
| Baseline (Llama 3.2)| 35.0%      | 19.1%      | 16.3%      |
| No Graph (Neo4j)    | 35.0%      | 22.0%      | -          |
| No Vector (FAISS)   | 28.0%      | 18.5%      | -          |
| No Vision (Mesh)    | 31.0%      | 25.0%      | -          |
| Full MTM Proposed   | 45.0%      | 40.5%      | 38.1%      |

---

## 15. Comparison with Prior Work

File: README.md

| Method (Citation)  | Accuracy (DocVQA) | Year |
|-------------------|------------|------|
| Llama 3.2 11B (Base)| 88.4%*     | 2024 |
| Qwen2-VL 7B        | 94.5%      | 2024 |
| GPT-4o-mini        | 92.8%      | 2024 |
| Claude 3.5 Sonnet  | 95.2%      | 2024 |
| MTM (Proposed)     | 45.0%**    | 2026 |

*Note: Base metrics are on full test sets; MTM metrics reflect internal multi-domain traceability test set.*

---

## 16. Chapter Config (optional overrides)

<!-- Ch.1 word range: 2000-3500 -->
<!-- Ch.2 word range: 4000-7000 -->
<!-- Ch.3 word range: 3500-6000 -->
<!-- Ch.4 word range: 4000-7000 -->
<!-- Ch.5 word range: 1500-2500 -->

---

## STATUS

Research Architecture:
- [x] Title confirmed
- [x] Problem written
- [x] Gap cited with >=3 papers
- [x] Aim written
- [x] RQ1/RQ2/RQ3 defined
- [x] Hypotheses stated
- [x] Method mapping complete
- [x] Advisor sign-off

Experiment Data:
- [x] Dataset stats filled
- [x] Experiment config filled
- [x] Primary results filled
- [x] Ablation/validation table filled
- [x] Comparison table filled

CHAPTER 4 BLOCKED until all experiment data items are checked.
