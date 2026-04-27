# MTM Evaluation Results Summary

This file provides a snapshot of the **Multimodal Traceability Mesh** performance metrics. For a detailed breakdown, see [EVALUATION_REPORT.md](./EVALUATION_REPORT.md).

---

## 🏆 Key Performance Gains
MTM transforms Llama 3.2 Vision into a grounded document analyst.

| Metric | Delta (FT vs Base) | Impact |
| :--- | :---: | :--- |
| **Hallucination (CHAIR-i)** | **-50.0%** | Massive reduction in false objects. |
| **Citation Accuracy** | **+133.7%** | Precision grounding in evidence. |
| **Factual Grounding (FActScore)**| **+112.0%** | Verifiable document analysis. |
| **Semantic Alignment (ROUGE-L)**| **+195.1%** | Adherence to structured protocols. |

---

## 🏎️ Efficiency & Scalability
*   **Mean Latency:** 2.49s (Optimized with vLLM KV-Cache Compression).
*   **OOD Detection:** 100% success rate in refusing questions outside document scope.
*   **Graph Logic:** 4x improvement in relationship extraction (Triple listing) through SFT.

---

## 🧩 Ablation Highlights
*   **The Neo4j Graph** is critical for spatial grounding (removing it drops FActScore by 18%).
*   **Vector Retrieval** ensures semantic coverage for non-visual queries.
*   **Llama 3.2 11B** remains the superior reasoning engine compared to Qwen2-VL 7B in this pipeline, despite slightly higher latency.

---
*Results curated on: Sunday, April 26, 2026*
