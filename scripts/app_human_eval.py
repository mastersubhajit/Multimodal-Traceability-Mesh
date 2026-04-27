"""
Streamlit human evaluation interface for VQA and provenance verification.
Supports per-sample annotation (fluency, faithfulness, citation, correctness, OOD),
inter-annotator agreement (Cohen's Kappa), and HIT template export.

Run: streamlit run scripts/app_human_eval.py
"""
import streamlit as st
import json
import os
import random
from datetime import datetime
from collections import defaultdict
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide", page_title="MTM Human Evaluation")
st.title("Multimodal Traceability Mesh — Human Evaluation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
SCORES_PATH = os.path.join(LOG_DIR, "human_eval_scores.csv")
HIT_TEMPLATE_PATH = os.path.join(LOG_DIR, "human_eval_hit_template.json")

ANNOTATION_DIMS = {
    "fluency":          ("Fluency [1–5]",           1, 5,  3,  "Grammatical fluency and readability"),
    "faithfulness":     ("Faithfulness [1–5]",       1, 5,  3,  "No hallucination; grounded in evidence"),
    "citation_quality": ("Citation Quality [0–2]",   0, 2,  0,  "0=none, 1=partial/wrong, 2=correct [n]"),
    "correctness":      ("Correctness [0–1]",        0, 1,  0,  "Final answer is correct"),
    "ood_refusal":      ("OOD Refusal [0–2]",        0, 2,  0,  "0=N/A, 1=refused no reason, 2=refused+reason"),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_results(path: str):
    with open(path) as f:
        data = json.load(f)
    results = data.get("details") or data.get("per_question_results") or data.get("results") or []
    return results


def cohen_kappa(scores1, scores2, field: str) -> float:
    v1 = [r.get(field) for r in scores1 if r.get(field) is not None]
    v2 = [r.get(field) for r in scores2 if r.get(field) is not None]
    if not v1 or len(v1) != len(v2):
        return float("nan")
    n = len(v1)
    agree = sum(a == b for a, b in zip(v1, v2)) / n
    from collections import Counter
    c1, c2 = Counter(v1), Counter(v2)
    labels = set(v1) | set(v2)
    expected = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    return (agree - expected) / (1.0 - expected) if expected != 1.0 else 1.0


def generate_hit_template():
    template = {
        "title":       "Evaluate AI Document Analysis Answers",
        "description": "Rate AI-generated answers on fluency, faithfulness, citation quality, and correctness.",
        "reward":      "$0.15 per HIT",
        "time_limit":  "10 minutes",
        "dimensions":  {k: v[0] for k, v in ANNOTATION_DIMS.items()},
        "example": {
            "question":   "What is the total score shown in the table?",
            "evidence":   "[1] The table shows a total score of 87/100.",
            "expected":   "87",
            "generated":  "Based on [1], the total score is 87 out of 100.",
            "ideal_ratings": {"fluency":5,"faithfulness":5,"citation_quality":2,"correctness":1,"ood_refusal":0},
        },
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(HIT_TEMPLATE_PATH, "w") as f:
        json.dump(template, f, indent=2)
    return template


# ---------------------------------------------------------------------------
# Sidebar — file selection
# ---------------------------------------------------------------------------
log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".json")] if os.path.isdir(LOG_DIR) else []
selected_log = st.sidebar.selectbox("Evaluation Log", log_files)
annotator_id = st.sidebar.text_input("Annotator ID", "A1")

tabs = st.tabs(["📝 Annotate", "📊 Summary & Kappa", "📄 HIT Template"])

# ===========================================================================
# Tab 1: Annotation
# ===========================================================================
with tabs[0]:
    if not selected_log:
        st.info("Select an evaluation log from the sidebar.")
    else:
        results = load_results(os.path.join(LOG_DIR, selected_log))
        if not results:
            st.error("No per-sample results found in this log.")
        else:
            st.sidebar.write(f"Total samples: {len(results)}")
            sample_idx = st.sidebar.number_input("Sample index", 0, len(results) - 1, 0)
            sample = results[sample_idx]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Input & Context")
                st.write(f"**Dataset:** {sample.get('dataset','?')}  |  **Category:** {sample.get('category','?')}")
                q = sample.get("question_text") or sample.get("question", "")
                st.write(f"**Question:** {q}")

                ev = sample.get("evidence", "")
                if ev:
                    with st.expander("Evidence blocks"):
                        st.write(ev[:800])

                img_path = sample.get("image_path")
                if img_path and os.path.exists(img_path):
                    st.image(Image.open(img_path), use_column_width=True)

            with col2:
                st.subheader("Model Output")
                st.write("**Expected:**")
                st.info(sample.get("expected") or sample.get("correct_answer", "N/A"))
                st.write("**Generated:**")
                st.success(sample.get("generated", "N/A"))
                reasoning = sample.get("reasoning", "")
                if reasoning:
                    st.write(f"**Reasoning:** {reasoning[:400]}")
                is_ood = sample.get("is_ood", False)
                if is_ood:
                    st.warning("⚠️ This sample was flagged as Out-of-Domain.")

                st.divider()
                st.subheader("Your Ratings")
                scores = {}
                for field, (label, lo, hi, default, tip) in ANNOTATION_DIMS.items():
                    scores[field] = st.slider(label, lo, hi, default, help=tip)
                notes = st.text_area("Notes (optional)")

                if st.button("💾 Save Rating"):
                    entry = {
                        "log":        selected_log,
                        "sample_idx": sample_idx,
                        "annotator":  annotator_id,
                        "timestamp":  datetime.now().isoformat(),
                        "dataset":    sample.get("dataset", ""),
                        "category":   sample.get("category", ""),
                        "notes":      notes,
                        **scores,
                    }
                    df_new = pd.DataFrame([entry])
                    if os.path.exists(SCORES_PATH):
                        df_new.to_csv(SCORES_PATH, mode="a", header=False, index=False)
                    else:
                        df_new.to_csv(SCORES_PATH, index=False)
                    st.success("Rating saved ✓")

# ===========================================================================
# Tab 2: Summary and Cohen's Kappa
# ===========================================================================
with tabs[1]:
    st.subheader("Annotation Summary")
    if not os.path.exists(SCORES_PATH):
        st.info("No annotations collected yet.")
    else:
        df = pd.read_csv(SCORES_PATH)
        st.write(f"Total annotations: {len(df)}")
        dims = list(ANNOTATION_DIMS.keys())
        mean_scores = df[dims].mean().rename("mean")
        st.dataframe(mean_scores.to_frame().T)

        st.subheader("Per-dataset / category breakdown")
        breakdown = df.groupby(["dataset", "category"])[dims].mean()
        st.dataframe(breakdown)

        st.subheader("Inter-Annotator Agreement (Cohen's κ)")
        annotators = df["annotator"].unique().tolist()
        if len(annotators) >= 2:
            a1_id = st.selectbox("Annotator 1", annotators, index=0)
            a2_id = st.selectbox("Annotator 2", annotators, index=min(1, len(annotators)-1))
            a1 = df[df["annotator"] == a1_id].to_dict("records")
            a2 = df[df["annotator"] == a2_id].to_dict("records")
            kappas = {}
            for field in dims:
                kappas[field] = cohen_kappa(a1, a2, field)
            st.dataframe(pd.DataFrame(kappas, index=["κ"]).T)
        else:
            st.info("Need at least 2 annotators for Kappa computation.")

# ===========================================================================
# Tab 3: HIT Template
# ===========================================================================
with tabs[2]:
    st.subheader("Crowdsourcing HIT Template")
    if st.button("Generate / Refresh HIT Template"):
        tmpl = generate_hit_template()
        st.success(f"Saved to {HIT_TEMPLATE_PATH}")

    if os.path.exists(HIT_TEMPLATE_PATH):
        with open(HIT_TEMPLATE_PATH) as f:
            tmpl = json.load(f)
        st.json(tmpl)
        st.download_button(
            "⬇️ Download HIT Template",
            data=json.dumps(tmpl, indent=2),
            file_name="hit_template.json",
            mime="application/json",
        )
    else:
        st.info("Click 'Generate HIT Template' above.")
