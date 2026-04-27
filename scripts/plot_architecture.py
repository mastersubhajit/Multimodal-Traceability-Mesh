"""
Publication-quality system architecture diagram for the MTM paper (Figure 1).
Generates figures/fig1_architecture.pdf + .png

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_architecture.py [--output_dir figures]
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Colors ────────────────────────────────────────────────────────────────────
C_STAGE   = "#2166AC"    # pipeline stage boxes
C_STORE   = "#4DAC26"    # storage (Neo4j / FAISS)
C_MODEL   = "#D6604D"    # model / inference boxes
C_INPUT   = "#762A83"    # input/output
C_ARROW   = "#444444"
C_DASHED  = "#888888"
C_BG      = "#F7F9FC"
C_TEXT    = "#FFFFFF"
C_DARK    = "#1A1A2E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def rounded_box(ax, x, y, w, h, color, text, fontsize=9.5, text_color="white",
                radius=0.04, zorder=4, alpha=1.0, bold=False):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle=f"round,pad=0.01,rounding_size={radius}",
                         linewidth=1.2, edgecolor="white",
                         facecolor=color, zorder=zorder, alpha=alpha)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight,
            zorder=zorder + 1, wrap=True,
            multialignment="center")
    return box


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5,
          style="->", dashed=False, zorder=3):
    ls = (0, (4, 3)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=style,
                    color=color,
                    lw=lw,
                    linestyle=ls,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=zorder)


def label_arrow(ax, x, y, text, fontsize=8.5, color="#555555"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            zorder=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fig_w, fig_h = 16, 9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor(C_BG)
    fig.patch.set_facecolor(C_BG)

    # ── Title ──────────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, 8.6, "Multimodal Traceability Mesh (MTM) — System Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK, zorder=10)

    # ── Row coordinates ────────────────────────────────────────────────────────
    #  Row 1 (top): Pipeline stages        y = 6.8
    #  Row 2:       Storage / retrievers   y = 4.8
    #  Row 3:       Inference paths        y = 2.8
    #  Row 4 (bot): Output                 y = 1.0

    BH = 0.80   # box height
    BW = 2.00   # box width (default)

    # ── ① INPUT ───────────────────────────────────────────────────────────────
    rounded_box(ax, 1.5, 6.8, BW, BH, C_INPUT,
                "① Document Input\n(PDF / Image)", fontsize=9, bold=True)

    # ── ② INGESTION ──────────────────────────────────────────────────────────
    rounded_box(ax, 4.2, 6.8, BW, BH, C_STAGE,
                "② Ingestion\nDocParser (OCR + Layout)", fontsize=9)
    arrow(ax, 2.51, 6.8, 3.19, 6.8)
    label_arrow(ax, 2.85, 6.98, "raw doc")

    # ── ③ MCQ DETECTION ───────────────────────────────────────────────────────
    rounded_box(ax, 6.9, 6.8, BW, BH, C_STAGE,
                "③ MCQ Detection\nStem + Option Parsing", fontsize=9)
    arrow(ax, 5.21, 6.8, 5.89, 6.8)
    label_arrow(ax, 5.55, 6.98, "blocks")

    # ── ④ GRAPH CONSTRUCTION ─────────────────────────────────────────────────
    rounded_box(ax, 9.6, 6.8, BW, BH, C_STAGE,
                "④ Graph Construction\nDocument→Page→Block", fontsize=9)
    arrow(ax, 7.91, 6.8, 8.59, 6.8)
    label_arrow(ax, 8.25, 6.98, "parsed")

    # ── ⑤ HYBRID RETRIEVAL ───────────────────────────────────────────────────
    rounded_box(ax, 12.3, 6.8, BW, BH, C_STAGE,
                "⑤ Hybrid Retrieval\nFAISS + CrossEncoder", fontsize=9)
    arrow(ax, 10.61, 6.8, 11.29, 6.8)
    label_arrow(ax, 10.95, 6.98, "graph")

    # ── ⑥ VISUAL MESH ─────────────────────────────────────────────────────────
    rounded_box(ax, 14.8, 6.8, BW + 0.2, BH, C_STAGE,
                "⑥ Visual Mesh\nLlama 3.2-11B Vision", fontsize=9)
    arrow(ax, 13.31, 6.8, 13.79, 6.8)
    label_arrow(ax, 13.55, 6.98, "top-k ctx")

    # ── STORAGE LAYER (row 2) ─────────────────────────────────────────────────

    # Neo4j
    rounded_box(ax, 9.6, 4.8, BW, BH, C_STORE,
                "Neo4j Graph DB\nDoc→Page→Block→Q", fontsize=9)
    # Arrow from Graph Construction down to Neo4j
    arrow(ax, 9.6, 6.39, 9.6, 5.21)
    label_arrow(ax, 9.82, 5.80, "store")

    # FAISS
    rounded_box(ax, 12.3, 4.8, BW, BH, C_STORE,
                "FAISS IndexFlatL2\nall-MiniLM-L6-v2", fontsize=9)
    arrow(ax, 12.3, 6.39, 12.3, 5.21)
    label_arrow(ax, 12.52, 5.80, "embed")

    # CrossEncoder reranker
    rounded_box(ax, 14.8, 4.8, BW + 0.2, BH, C_STORE,
                "CrossEncoder Reranker\nms-marco-MiniLM-L-6", fontsize=9)
    # Neo4j feeds into retrieval (horizontal)
    arrow(ax, 10.61, 4.8, 11.29, 4.8, style="->" )
    label_arrow(ax, 10.95, 4.98, "graph ctx")
    # FAISS → CrossEncoder
    arrow(ax, 13.31, 4.8, 13.79, 4.8, style="->")
    label_arrow(ax, 13.55, 4.98, "top-k")

    # ── INFERENCE PATHS (row 3) ───────────────────────────────────────────────
    # Dashed separator line
    ax.axhline(3.85, xmin=0.02, xmax=0.98, color="#BBBBBB", lw=1.0,
               linestyle="--", zorder=2)
    ax.text(0.3, 3.72, "Evaluation Paths:", fontsize=9, color="#666666",
            style="italic")

    # Base model path box
    rounded_box(ax, 4.2, 2.8, 3.6, BH, C_MODEL,
                "Base Model Path\n(No graph context)\nImage + Question only",
                fontsize=9, alpha=0.90)

    # RAG-adapted path box
    rounded_box(ax, 10.5, 2.8, 3.8, BH, C_MODEL,
                "RAG-Adapted (MTM) Path\nImage + Numbered Evidence\n+ Citation Instruction",
                fontsize=9, alpha=0.90)

    # Arrows from Visual Mesh down to both paths
    arrow(ax, 14.8, 6.39, 14.8, 3.5, dashed=False)
    arrow(ax, 14.8, 3.5, 10.5, 3.21)
    label_arrow(ax, 12.65, 3.6, "RAG")
    arrow(ax, 14.8, 3.5,  4.2, 3.21, dashed=True)
    label_arrow(ax, 9.5, 3.42, "Base (image only)")

    # QLoRA / PEFT tag
    rounded_box(ax, 10.5, 1.95, 2.6, 0.52, "#9C3A2C",
                "QLoRA Fine-tuning (rank=64, α=16)\n4-bit NF4 quantization",
                fontsize=8, alpha=0.90)
    arrow(ax, 10.5, 2.38, 10.5, 2.22)

    # ── OOD CHECK & VERIFICATION ──────────────────────────────────────────────
    rounded_box(ax, 4.2, 1.6, 3.6, 0.65, "#1B7837",
                "OOD Detection\n\"In the document, there is no mention of...\"",
                fontsize=8.5, bold=False)
    rounded_box(ax, 10.5, 1.3, 3.8, 0.65, "#1B7837",
                "Inference + Verification\nAnswer + [n] Citations + Confidence",
                fontsize=8.5, bold=False)

    arrow(ax, 4.2, 2.38, 4.2, 1.93)
    arrow(ax, 10.5, 2.38, 10.5, 1.63)

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    rounded_box(ax, 7.35, 0.52, 5.2, 0.62, C_INPUT,
                "Output: Grounded Answer | Evidence Traceability | Hallucination Score",
                fontsize=9, bold=True)
    arrow(ax, 4.2, 1.26, 4.2, 0.52)
    arrow(ax, 4.2, 0.52, 4.74, 0.52)
    arrow(ax, 10.5, 0.97, 10.5, 0.52)
    arrow(ax, 10.5, 0.52, 9.96, 0.52)

    # ── LEGEND ────────────────────────────────────────────────────────────────
    legend_items = [
        (C_STAGE,  "Pipeline Stage"),
        (C_STORE,  "Storage / Retriever"),
        (C_MODEL,  "Inference Path"),
        (C_INPUT,  "Input / Output"),
        (C_GOOD := "#1B7837", "Verification"),
    ]
    patches = [mpatches.Patch(facecolor=c, label=l, edgecolor="white")
               for c, l in legend_items]
    ax.legend(handles=patches, loc="lower left",
              bbox_to_anchor=(0.01, 0.01),
              fontsize=9, framealpha=0.9,
              ncol=len(legend_items))

    # ── Save ──────────────────────────────────────────────────────────────────
    stem = "fig1_architecture"
    for ext in ("pdf", "png"):
        path = os.path.join(args.output_dir, f"{stem}.{ext}")
        fig.savefig(path, facecolor=C_BG)
    plt.close(fig)
    print(f"  Saved  {stem}.pdf / .png  →  {args.output_dir}/")


if __name__ == "__main__":
    main()
