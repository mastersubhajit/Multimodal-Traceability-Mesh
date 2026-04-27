"""
Publication-quality figures for the Multimodal Traceability Mesh (MTM) paper.
Generates all evaluation figures as PDF + PNG under figures/.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_results.py [--output_dir figures]
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

# Color palette
C_BASE  = "#2166AC"   # deep blue  — Base model
C_RAG   = "#D6604D"   # coral red  — RAG / FT model
C_GOOD  = "#4DAC26"   # green      — positive direction
C_GRAY  = "#878787"   # mid gray   — neutral
C_GOLD  = "#F4A582"   # pale orange
PALETTE = [C_BASE, C_RAG, C_GOOD, C_GOLD, C_GRAY, "#762A83", "#1B7837", "#C2A5CF"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def save(fig, path_stem: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{path_stem}.{ext}"))
    plt.close(fig)
    print(f"  Saved  {path_stem}.pdf / .png")


def _bar_group(ax, labels, vals_list, group_labels, colors, width=0.35,
               ylim=None, ylabel="Score", title="", annotation=True):
    x = np.arange(len(labels))
    n = len(vals_list)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)
    bars_all = []
    for i, (vals, col) in enumerate(zip(vals_list, colors)):
        b = ax.bar(x + offsets[i], vals, width, label=group_labels[i],
                   color=col, alpha=0.88, zorder=3)
        bars_all.append(b)
        if annotation:
            for rect in b:
                h = rect.get_height()
                if h > 0.01:
                    ax.annotate(f"{h:.2f}",
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontweight="bold", pad=6)
    ax.legend(framealpha=0.9)
    return bars_all


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_all(log_dir: str):
    d = {}
    for key, fname in [
        ("base",       "eval_base_results.json"),
        ("ft",         "eval_ft_results.json"),
        ("graph_base", "eval_graph_base.json"),
        ("graph_ft",   "eval_graph_ft.json"),
        ("vis_base",   "eval_vision_base.json"),
        ("vis_ft",     "eval_vision_ft.json"),
        ("ablation",   "ablation_study.json"),
        ("err_base",   "eval_base_results_error_analysis.json"),
        ("comprehensive", "comprehensive_eval_report.json"),
        ("graph_cmp",  "graph_ft_comparison.json"),
    ]:
        d[key] = load_json(os.path.join(log_dir, fname))
    return d


# ── Figure 2: Overall Metrics ─────────────────────────────────────────────────

def fig_overall_metrics(D, out_dir):
    base_ov = D["base"]["overall"]  if D["base"]  else {}
    ft_ov   = D["ft"]["overall"]    if D["ft"]    else {}

    # Prefer comprehensive report when available
    if D["comprehensive"]:
        c = D["comprehensive"]
        base_ov = c["overall"]["base"] if "overall" in c else base_ov
        ft_ov   = c["overall"]["rag"]  if "overall" in c else ft_ov

    metrics = [
        ("Accuracy",        "accuracy"),
        ("Hit Rate",        "hit_rate_at1"),
        ("MRR",             "mrr"),
        ("ROUGE-L",         "rougeL"),
        ("FActScore",       "factscore"),
        ("Citation Acc.",   "citation"),
        ("BLEU",            "bleu"),
    ]
    labels = [m[0] for m in metrics]

    def _get(d, keys):
        for k in keys:
            if k in d:
                return float(d[k])
        return 0.0

    base_vals = [_get(base_ov, [m[1], m[1].lower()]) for m in metrics]
    ft_vals   = [_get(ft_ov,   [m[1], m[1].lower()]) for m in metrics]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar_group(ax, labels, [base_vals, ft_vals],
               ["Base (Llama-3.2-11B)", "RAG-Adapted (MTM)"],
               [C_BASE, C_RAG], width=0.35,
               ylim=(0, 1.0), ylabel="Score",
               title="Overall Performance: Base vs. RAG-Adapted MTM")
    ax.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
    fig.tight_layout()
    save(fig, "fig2_overall_metrics", out_dir)


# ── Figure 3: Per-Dataset Performance ────────────────────────────────────────

def fig_per_dataset(D, out_dir):
    base_ds = D["base"]["per_dataset"] if D["base"] else {}
    ft_ds   = D["ft"]["per_dataset"]   if D["ft"]   else {}

    datasets = sorted(set(list(base_ds.keys()) + list(ft_ds.keys())))
    if not datasets:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, (metric, label) in zip(axes, [
        ("accuracy",  "Accuracy"),
        ("rougeL",    "ROUGE-L"),
        ("factscore", "FActScore"),
    ]):
        base_v = [base_ds.get(ds, {}).get(metric, 0.0) for ds in datasets]
        ft_v   = [ft_ds.get(ds,   {}).get(metric, 0.0) for ds in datasets]
        _bar_group(ax, datasets, [base_v, ft_v],
                   ["Base", "RAG-Adapted"],
                   [C_BASE, C_RAG], width=0.35,
                   ylim=(0, 1.0), ylabel=label, title=label)

    fig.suptitle("Per-Dataset Performance", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_per_dataset", out_dir)


# ── Figure 4: Document-Category Breakdown ────────────────────────────────────

def fig_docvqa_categories(D, out_dir):
    base_cat = D["base"].get("per_category", {}) if D["base"] else {}
    ft_cat   = D["ft"].get("per_category",   {}) if D["ft"]   else {}

    # Keep only DocVQA categories
    cats = sorted(set(
        list(base_cat.keys()) + list(ft_cat.keys())
    ))
    if not cats:
        return

    def short(c):
        return c.split("/")[-1].replace("_", " ").title()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (metric, label) in zip(axes, [("accuracy", "Accuracy"), ("rougeL", "ROUGE-L")]):
        b_vals = [base_cat.get(c, {}).get(metric, 0) for c in cats]
        f_vals = [ft_cat.get(c,   {}).get(metric, 0) for c in cats]
        short_cats = [short(c) for c in cats]
        _bar_group(ax, short_cats, [b_vals, f_vals],
                   ["Base", "RAG-Adapted"], [C_BASE, C_RAG],
                   width=0.35, ylim=(0, 1.0), ylabel=label,
                   title=f"{label} by Category", annotation=False)

    fig.suptitle("Question-Category Breakdown", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_category_breakdown", out_dir)


# ── Figure 5: Hallucination Metrics ──────────────────────────────────────────

def fig_hallucination(D, out_dir):
    vb = D["vis_base"] or {}
    vf = D["vis_ft"]   or {}

    # Also pull from comprehensive if available
    if D["comprehensive"] and "hallucination" in D["comprehensive"]:
        hall_c = D["comprehensive"]["hallucination"]
        if "base" in hall_c and hall_c["base"]:
            vb = hall_c["base"]
        if "rag" in hall_c and hall_c["rag"]:
            vf = hall_c["rag"]

    def g(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                v = d[k]
                if isinstance(v, dict):
                    for kk in ("chair_i_mean", "chair_s_mean", "caos_mean",
                               "accuracy", "hallucination_rate",
                               "refusal_rate", "reasoning_rate"):
                        if kk in v:
                            return float(v[kk])
                return float(v)
        return 0.0

    panel_data = {
        "CHAIR-i\n(↓ better)": (
            g(vb, "CHAIR", "chair_i_mean"), g(vf, "CHAIR", "chair_i_mean"), True
        ),
        "CHAIR-s\n(↓ better)": (
            g(vb, "CHAIR", "chair_s_mean"), g(vf, "CHAIR", "chair_s_mean"), True
        ),
        "CAOS\n(↑ better)": (
            g(vb, "CAOS", "caos_mean"), g(vf, "CAOS", "caos_mean"), False
        ),
        "NOPE Acc.\n(↑ better)": (
            g(vb, "NOPE", "accuracy"),  g(vf, "NOPE", "accuracy"),  False
        ),
        "I-HallA Acc.\n(↑ better)": (
            g(vb, "I_HallA", "accuracy"), g(vf, "I_HallA", "accuracy"), False
        ),
        "OOD Refusal\n(↑ better)": (
            g(vb, "OOD_Refusal", "refusal_rate"),
            g(vf, "OOD_Refusal", "refusal_rate"), False
        ),
    }

    keys   = list(panel_data.keys())
    b_vals = [panel_data[k][0] for k in keys]
    f_vals = [panel_data[k][1] for k in keys]
    lower_better = [panel_data[k][2] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(keys))
    w = 0.35
    b_bars = ax.bar(x - w/2, b_vals, w, label="Base",        color=C_BASE, alpha=0.88, zorder=3)
    f_bars = ax.bar(x + w/2, f_vals, w, label="RAG-Adapted", color=C_RAG,  alpha=0.88, zorder=3)

    for bars in (b_bars, f_bars):
        for rect in bars:
            h = rect.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5)

    # Highlight lower-is-better panels with subtle shade
    for i, lb in enumerate(lower_better):
        if lb:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06, color="#999999", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Hallucination & Robustness Metrics", fontweight="bold", pad=8)
    ax.legend(framealpha=0.9)

    note = ax.text(0.99, 0.97, "Shaded: lower is better",
                   transform=ax.transAxes, ha="right", va="top",
                   fontsize=8.5, color="#666666",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))
    fig.tight_layout()
    save(fig, "fig5_hallucination", out_dir)


# ── Figure 6: Graph Comprehension Before/After Finetuning ────────────────────

def fig_graph_comprehension(D, out_dir):
    TASK_TYPES = [
        "N. number", "E. number", "Triple listing",
        "N. degree", "Highest N. degree", "N. description",
    ]

    def compute_acc(data):
        if not data:
            return {t: 0.0 for t in TASK_TYPES}
        if isinstance(data, list):
            acc = {}
            for t in TASK_TYPES:
                sub = [r for r in data if r.get("type") == t]
                acc[t] = (sum(1 for r in sub if r.get("is_correct")) / len(sub)
                          if sub else 0.0)
            return acc
        # graph_ft_comparison format
        if "comparison" in data:
            return {t: data["comparison"].get(t, {}).get("finetuned", 0.0)
                    for t in TASK_TYPES}
        return {t: 0.0 for t in TASK_TYPES}

    def compute_base_acc(data):
        if not data:
            return {t: 0.0 for t in TASK_TYPES}
        if isinstance(data, list):
            acc = {}
            for t in TASK_TYPES:
                sub = [r for r in data if r.get("type") == t]
                acc[t] = (sum(1 for r in sub if r.get("is_correct")) / len(sub)
                          if sub else 0.0)
            return acc
        if "comparison" in data:
            return {t: data["comparison"].get(t, {}).get("base", 0.0)
                    for t in TASK_TYPES}
        return {t: 0.0 for t in TASK_TYPES}

    # Prefer graph_cmp (multi-doc) over single-doc graph_base/graph_ft
    if D["graph_cmp"]:
        base_acc = compute_base_acc(D["graph_cmp"])
        ft_acc   = compute_acc(D["graph_cmp"])
    else:
        base_acc = compute_acc(D["graph_base"])
        ft_acc   = compute_acc(D["graph_ft"])

    short_labels = ["N. Number", "E. Number", "Triple List.",
                    "N. Degree", "Max Degree", "N. Desc."]
    b_vals = [base_acc[t] for t in TASK_TYPES]
    f_vals = [ft_acc[t]   for t in TASK_TYPES]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar_group(ax, short_labels, [b_vals, f_vals],
               ["Base Model", "Graph-Finetuned"],
               [C_BASE, C_RAG], width=0.35,
               ylim=(0, 1.15), ylabel="Accuracy",
               title="Graph Comprehension: Base vs. Graph-Finetuned Model")

    # Delta annotations
    x = np.arange(len(TASK_TYPES))
    for i, (bv, fv) in enumerate(zip(b_vals, f_vals)):
        delta = fv - bv
        if abs(delta) > 0.005:
            color = C_GOOD if delta > 0 else "#C0392B"
            ax.annotate(f"Δ{delta:+.2f}",
                        xy=(x[i], max(bv, fv) + 0.07),
                        ha="center", fontsize=8, color=color, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig6_graph_comprehension", out_dir)


# ── Figure 7: Ablation Study ──────────────────────────────────────────────────

def fig_ablation(D, out_dir):
    if not D["ablation"]:
        return
    abl = D["ablation"]

    comp = abl.get("component_ablation", {})
    model_abl = abl.get("model_ablation", {})

    labels_comp = {
        "full_pipeline":      "Full MTM",
        "no_graph":           "w/o Graph",
        "no_vector":          "w/o Vector",
        "no_vision":          "w/o Vision",
        "no_graph_no_vector": "w/o Graph+Vec",
        "vanilla_vlm":        "Vanilla VLM",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Component ablation — accuracy + latency
    keys_c = [k for k in labels_comp if k in comp]
    acc_c  = [comp[k].get("percentage", 0.0) / 100.0 for k in keys_c]
    lat_c  = [comp[k].get("elapsed", 0.0) / max(comp[k].get("total", 1), 1)
               for k in keys_c]
    xlabels_c = [labels_comp[k] for k in keys_c]

    ax = axes[0]
    colors_c = [C_RAG if k == "full_pipeline" else C_GRAY for k in keys_c]
    bars = ax.bar(xlabels_c, acc_c, color=colors_c, alpha=0.88, zorder=3,
                  edgecolor="white", linewidth=0.5)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Component Ablation — Accuracy", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)

    # Latency per component
    ax2 = ax.twinx()
    ax2.plot(xlabels_c, lat_c, marker="o", color=C_GOLD, lw=2, label="Latency (s/item)", zorder=4)
    ax2.set_ylabel("Latency (s / item)", color=C_GOLD)
    ax2.tick_params(axis="y", labelcolor=C_GOLD)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    lines, lbels = ax2.get_legend_handles_labels()
    ax.legend([mpatches.Patch(color=C_RAG), mpatches.Patch(color=C_GRAY)] + lines,
              ["Full MTM", "Ablated"] + lbels, fontsize=9)

    # Model ablation
    ax = axes[1]
    model_keys   = list(model_abl.keys())
    model_acc    = [model_abl[k].get("percentage", 0.0) / 100.0 for k in model_keys]
    model_lat    = [model_abl[k].get("elapsed", 0.0) / max(model_abl[k].get("total", 1), 1)
                    for k in model_keys]
    short_models = [k.split("/")[-1] if "/" in k else k for k in model_keys]
    bars = ax.bar(short_models, model_acc, color=[C_BASE, C_GOLD], alpha=0.88,
                  zorder=3, edgecolor="white")
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax3 = ax.twinx()
    ax3.plot(short_models, model_lat, marker="s", color=C_GOOD, lw=2, label="Latency (s/item)")
    ax3.set_ylabel("Latency (s / item)", color=C_GOOD)
    ax3.tick_params(axis="y", labelcolor=C_GOOD)
    ax3.spines["right"].set_visible(True)
    ax3.spines["top"].set_visible(False)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Ablation — Accuracy & Latency", fontweight="bold")

    fig.suptitle("Ablation Study", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig7_ablation", out_dir)


# ── Figure 8: OOD Detection ───────────────────────────────────────────────────

def fig_ood_detection(D, out_dir):
    # Pull from comprehensive if available, else vision reports
    if D["comprehensive"] and "ood_detection" in D["comprehensive"]:
        ood_base = D["comprehensive"]["ood_detection"].get("base", {})
        ood_rag  = D["comprehensive"]["ood_detection"].get("rag",  {})
    else:
        vb = D["vis_base"] or {}
        vf = D["vis_ft"]   or {}
        ood_base = vb.get("OOD_Refusal", {})
        ood_rag  = vf.get("OOD_Refusal", {})

    labels   = ["Refusal Rate\n(↑ better)", "Reasoning Rate\n(↑ better)"]
    b_vals   = [ood_base.get("refusal_rate", 0.0), ood_base.get("reasoning_rate", 0.0)]
    r_vals   = [ood_rag.get("refusal_rate",  0.0), ood_rag.get("reasoning_rate",  0.0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_group(ax, labels, [b_vals, r_vals],
               ["Base Model", "RAG-Adapted"],
               [C_BASE, C_RAG], width=0.3,
               ylim=(0, 1.15), ylabel="Rate",
               title="Out-of-Domain (OOD) Refusal Performance")
    ax.axhline(1.0, color="#555555", ls=":", lw=0.8, alpha=0.5)
    fig.tight_layout()
    save(fig, "fig8_ood_detection", out_dir)


# ── Figure 9: Error Analysis ──────────────────────────────────────────────────

def fig_error_analysis(D, out_dir):
    err = D["err_base"]
    if not err:
        return

    cats = err.get("categories", {})
    non_zero = {k: v for k, v in cats.items() if isinstance(v, dict) and v.get("count", 0) > 0}
    if not non_zero:
        # All categories zero — show per_dataset_category errors
        pde = err.get("per_dataset_category_errors", {})
        if not pde:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        all_ds = list(pde.keys())
        all_cats_set = sorted(set(c for v in pde.values() for c in v))
        bottom = np.zeros(len(all_ds))
        for ci, cat in enumerate(all_cats_set):
            vals = [pde[ds].get(cat, 0) for ds in all_ds]
            ax.bar(all_ds, vals, bottom=bottom,
                   label=cat, color=PALETTE[ci % len(PALETTE)], alpha=0.88)
            bottom += np.array(vals)
        ax.set_ylabel("Error Count")
        ax.set_title("Errors by Dataset & Category", fontweight="bold")
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        save(fig, "fig9_error_analysis", out_dir)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Pie chart
    ax = axes[0]
    labels_pie = list(non_zero.keys())
    sizes  = [non_zero[k]["count"] for k in labels_pie]
    wedge_colors = PALETTE[:len(labels_pie)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, [l.replace("_", " ") for l in labels_pie],
              loc="lower center", bbox_to_anchor=(0.5, -0.18),
              fontsize=9, ncol=2)
    ax.set_title("Error Category Distribution\n(Base Model)", fontweight="bold")

    # Bar chart by dataset/category
    pde = err.get("per_dataset_category_errors", {})
    ax = axes[1]
    if pde:
        all_ds  = list(pde.keys())
        all_sub = sorted(set(c for v in pde.values() for c in v))
        bottom  = np.zeros(len(all_ds))
        for ci, subcat in enumerate(all_sub):
            vals = [pde[ds].get(subcat, 0) for ds in all_ds]
            ax.bar(all_ds, vals, bottom=bottom,
                   label=subcat, color=PALETTE[ci % len(PALETTE)], alpha=0.88)
            bottom += np.array(vals)
        ax.set_ylabel("Error Count")
        ax.set_title("Errors per Dataset / Sub-Category", fontweight="bold")
        ax.legend(fontsize=8.5, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Error Analysis", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig9_error_analysis", out_dir)


# ── Figure 10: Generation Quality (ROUGE-L / BLEU / FActScore) ───────────────

def fig_generation_quality(D, out_dir):
    base_ds = D["base"]["per_dataset"] if D["base"] else {}
    ft_ds   = D["ft"]["per_dataset"]   if D["ft"]   else {}
    datasets = sorted(set(list(base_ds.keys()) + list(ft_ds.keys())))
    if not datasets:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (metric, label) in zip(axes, [
        ("rougeL",    "ROUGE-L"),
        ("bleu",      "BLEU"),
        ("factscore", "FActScore"),
    ]):
        b_v = [base_ds.get(ds, {}).get(metric, 0) for ds in datasets]
        f_v = [ft_ds.get(ds,   {}).get(metric, 0) for ds in datasets]
        _bar_group(ax, datasets, [b_v, f_v],
                   ["Base", "RAG-Adapted"], [C_BASE, C_RAG],
                   width=0.35, ylim=(0, 1.0), ylabel=label, title=label)
    fig.suptitle("Generation Quality per Dataset", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig10_generation_quality", out_dir)


# ── Figure 11: Latency Analysis ───────────────────────────────────────────────

def fig_latency(D, out_dir):
    base_ds = D["base"]["per_dataset"] if D["base"] else {}
    ft_ds   = D["ft"]["per_dataset"]   if D["ft"]   else {}

    base_lat_ov = D["base"].get("latency", {}) if D["base"] else {}
    ft_lat_ov   = D["ft"].get("latency",   {}) if D["ft"]   else {}

    datasets = sorted(set(list(base_ds.keys()) + list(ft_ds.keys())))
    if not datasets:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Per-dataset latency
    ax = axes[0]
    b_lat = [base_ds.get(ds, {}).get("latency_mean", 0) for ds in datasets]
    f_lat = [ft_ds.get(ds,   {}).get("latency_mean", 0) for ds in datasets]
    _bar_group(ax, datasets, [b_lat, f_lat],
               ["Base", "RAG-Adapted"], [C_BASE, C_RAG],
               width=0.35, ylabel="Latency (s / query)", title="Per-Dataset Inference Latency")

    # Overall latency with std
    ax = axes[1]
    models  = ["Base", "RAG-Adapted"]
    means   = [base_lat_ov.get("mean", 0), ft_lat_ov.get("mean", 0)]
    stds    = [base_lat_ov.get("std",  0), ft_lat_ov.get("std",  0)]
    bars = ax.bar(models, means, color=[C_BASE, C_RAG], alpha=0.88,
                  yerr=stds, capsize=6, zorder=3,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}s",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean Latency (s ± std)")
    ax.set_title("Overall Inference Latency (Mean ± Std)", fontweight="bold")

    fig.suptitle("Latency Analysis", fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig11_latency", out_dir)


# ── Figure 12: Radar / Spider Chart — Multi-Metric Summary ───────────────────

def fig_radar_summary(D, out_dir):
    base_ov = D["base"]["overall"] if D["base"] else {}
    ft_ov   = D["ft"]["overall"]   if D["ft"]   else {}

    if D["comprehensive"]:
        c = D["comprehensive"]
        base_ov = c["overall"].get("base", base_ov)
        ft_ov   = c["overall"].get("rag",  ft_ov)

    categories = ["Accuracy", "MRR", "ROUGE-L", "FActScore", "Citation", "BLEU"]
    key_map    = ["accuracy", "mrr", "rougeL", "factscore", "citation", "bleu"]

    def _get(d, k):
        return float(d.get(k, 0.0))

    base_vals = [_get(base_ov, k) for k in key_map]
    ft_vals   = [_get(ft_ov,   k) for k in key_map]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    base_vals_c = base_vals + [base_vals[0]]
    ft_vals_c   = ft_vals   + [ft_vals[0]]
    angles_c    = angles    + [angles[0]]
    cats_c      = categories + [categories[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles_c, base_vals_c, "o-", lw=2, color=C_BASE, label="Base")
    ax.fill(angles_c, base_vals_c, alpha=0.15, color=C_BASE)
    ax.plot(angles_c, ft_vals_c,   "s-", lw=2, color=C_RAG,  label="RAG-Adapted")
    ax.fill(angles_c, ft_vals_c,   alpha=0.15, color=C_RAG)

    ax.set_thetagrids(np.degrees(angles), categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666666")
    ax.set_title("Multi-Metric Radar Summary", fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    save(fig, "fig12_radar_summary", out_dir)


# ── Figure 13: CSQA / OBQA Benchmarks ────────────────────────────────────────

def fig_csqa_obqa(D, out_dir):
    if not D["comprehensive"]:
        return
    c = D["comprehensive"]
    csqa_b = c.get("csqa", {}).get("base", {})
    csqa_r = c.get("csqa", {}).get("rag",  {})
    obqa_b = c.get("obqa", {}).get("base", {})
    obqa_r = c.get("obqa", {}).get("rag",  {})

    if not any([csqa_b, csqa_r, obqa_b, obqa_r]):
        return

    benchmarks = ["CSQA", "OBQA"]
    base_acc = [csqa_b.get("accuracy", 0), obqa_b.get("accuracy", 0)]
    rag_acc  = [csqa_r.get("accuracy", 0), obqa_r.get("accuracy", 0)]
    ns       = [csqa_b.get("total", 0), obqa_b.get("total", 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    _bar_group(ax, [f"{b}\n(n={n})" for b, n in zip(benchmarks, ns)],
               [base_acc, rag_acc],
               ["Base", "RAG-Adapted"], [C_BASE, C_RAG],
               width=0.3, ylim=(0, 1.0), ylabel="Accuracy",
               title="Commonsense & OpenBook QA Benchmarks")
    fig.tight_layout()
    save(fig, "fig13_csqa_obqa", out_dir)


# ── Figure 14: Comprehensive split/dataset heatmap ───────────────────────────

def fig_heatmap(D, out_dir):
    base_ds = D["base"]["per_dataset"] if D["base"] else {}
    ft_ds   = D["ft"]["per_dataset"]   if D["ft"]   else {}
    datasets = sorted(set(list(base_ds.keys()) + list(ft_ds.keys())))
    if not datasets:
        return

    metrics = ["accuracy", "rougeL", "factscore", "citation", "mrr"]
    m_labels = ["Accuracy", "ROUGE-L", "FActScore", "Citation", "MRR"]

    base_mat = np.array([[base_ds.get(d, {}).get(m, 0) for m in metrics] for d in datasets])
    ft_mat   = np.array([[ft_ds.get(d,   {}).get(m, 0) for m in metrics] for d in datasets])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(datasets) * 0.75 + 1.8)),
                             gridspec_kw={"wspace": 0.35})
    for ax, mat, title in [
        (axes[0], base_mat, "Base Model"),
        (axes[1], ft_mat,   "RAG-Adapted (MTM)"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(m_labels, rotation=35, ha="right", fontsize=9.5)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=10)
        ax.tick_params(axis="x", pad=2)
        for i in range(len(datasets)):
            for j in range(len(metrics)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=9, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03, shrink=0.85)
    cbar.set_label("Score", fontsize=10)
    fig.suptitle("Performance Heatmap — Dataset × Metric", fontweight="bold", fontsize=13, y=1.02)
    fig.savefig(os.path.join(out_dir, "fig14_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig14_heatmap.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  Saved  fig14_heatmap.pdf / .png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir",    default="logs")
    ap.add_argument("--output_dir", default="figures")
    args = ap.parse_args()

    print(f"\nLoading data from {args.log_dir} …")
    D = load_all(args.log_dir)

    figs = [
        ("fig2_overall_metrics",      fig_overall_metrics),
        ("fig3_per_dataset",           fig_per_dataset),
        ("fig4_category_breakdown",    fig_docvqa_categories),
        ("fig5_hallucination",         fig_hallucination),
        ("fig6_graph_comprehension",   fig_graph_comprehension),
        ("fig7_ablation",              fig_ablation),
        ("fig8_ood_detection",         fig_ood_detection),
        ("fig9_error_analysis",        fig_error_analysis),
        ("fig10_generation_quality",   fig_generation_quality),
        ("fig11_latency",              fig_latency),
        ("fig12_radar_summary",        fig_radar_summary),
        ("fig13_csqa_obqa",            fig_csqa_obqa),
        ("fig14_heatmap",              fig_heatmap),
    ]

    print(f"\nGenerating {len(figs)} figures → {args.output_dir}/\n")
    for name, fn in figs:
        try:
            fn(D, args.output_dir)
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")

    print(f"\nDone. Figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
