"""Generate all 5 publication-quality figures for the SEG paper."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (7, 5),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Color palette
C_SPECIALIST = "#82b366"   # green
C_GENERALIST = "#d79b00"   # orange
C_BASELINE = "#6c8ebf"     # blue
C_OURS = "#b85450"         # red
C_CPU = "#9673a6"          # purple
C_GPU = "#d6b656"          # yellow


def fig1_pipeline_diagram():
    """Figure 1: Pipeline architecture as a flowchart-style matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("SGAF Pipeline Architecture (Frozen B5 + P3)", fontsize=14, fontweight="bold", pad=20)

    def draw_box(x, y, w, h, text, color, fontsize=9, fontweight="normal"):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, wrap=True)

    def draw_arrow(x1, y1, x2, y2, label="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if label:
            mid_x, mid_y = (x1 + x2) / 2 + 0.15, (y1 + y2) / 2
            ax.text(mid_x, mid_y, label, fontsize=7, ha="left", va="center", style="italic")

    # Input
    draw_box(5, 7.2, 2.8, 0.6, "Query Batch", C_BASELINE, fontweight="bold")

    # Retrievers
    draw_box(2, 5.5, 2.4, 0.7, "BGE-small (Specialist)\n33M, 384d", C_SPECIALIST, fontsize=8, fontweight="bold")
    draw_box(5, 5.5, 2.4, 0.7, "BGE-base (Generalist)\n109M, 768d", C_GENERALIST, fontsize=8, fontweight="bold")
    draw_box(8, 5.5, 2.0, 0.7, "BM25\n(Lexical Baseline)", C_BASELINE, fontsize=8)

    # Features
    draw_box(5, 4.0, 3.2, 0.7, "Extract 5 Query Features\n(z-score: gap, std, overlap, top, len)", C_CPU, fontsize=8)

    # B5 Decision
    draw_box(5, 2.5, 3.0, 0.8, "B5 Batch Shift Score S ≥ 2.0?", C_OURS, fontsize=9, fontweight="bold")

    # Safe / Fallback
    draw_box(1.8, 1.0, 2.4, 0.7, "Specialist-Safe\n(BGE-small ranking)", C_SPECIALIST, fontsize=8, fontweight="bold")
    draw_box(5.0, 1.0, 2.6, 0.7, "Generalist-Fallback\n(BGE-base ranking)", C_GENERALIST, fontsize=8, fontweight="bold")

    # P3
    draw_box(7.8, 1.0, 2.2, 0.7, "P3 Smoothing\n(Top-20 RRF blend\nα=0.1, CPU)", C_CPU, fontsize=7)

    # Output
    draw_box(5, -0.3, 2.8, 0.6, "Final Fused Ranking", C_BASELINE, fontsize=9, fontweight="bold")

    # Arrows
    draw_arrow(5, 6.85, 2, 5.85)
    draw_arrow(5, 6.85, 5, 5.85)
    draw_arrow(5, 6.85, 8, 5.85)
    draw_arrow(2, 5.15, 5, 4.35)
    draw_arrow(5, 5.15, 5, 4.35)
    draw_arrow(5, 3.65, 5, 2.9)

    # Decision branches
    ax.annotate("", xy=(1.8, 1.35), xytext=(3.5, 2.1),
        arrowprops=dict(arrowstyle="->", color=C_SPECIALIST, lw=1.5))
    ax.text(2.2, 2.3, "NO (S<2.0)", fontsize=7, color=C_SPECIALIST, fontweight="bold")

    ax.annotate("", xy=(5, 1.35), xytext=(5, 2.1),
        arrowprops=dict(arrowstyle="->", color=C_GENERALIST, lw=1.5))
    ax.text(5.3, 2.3, "YES (S≥2.0)", fontsize=7, color=C_GENERALIST, fontweight="bold")

    draw_arrow(5, 0.65, 7.8, 0.65)

    # All converge to output
    draw_arrow(1.8, 0.65, 5, 0.05, color=C_SPECIALIST)
    draw_arrow(7.8, 0.65, 6.4, 0.05, color=C_CPU)

    # Legend
    legend_elements = [
        mpatches.Patch(color=C_SPECIALIST, label="BGE-small Specialist"),
        mpatches.Patch(color=C_GENERALIST, label="BGE-base Generalist"),
        mpatches.Patch(color=C_CPU, label="CPU-only (Post-Retrieval)"),
        mpatches.Patch(color=C_OURS, label="B5 Decision Gate"),
        mpatches.Patch(color=C_BASELINE, label="Input / Output"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=7,
             bbox_to_anchor=(0.5, -0.18))

    fig.savefig(FIG_DIR / "fig1_pipeline.pdf")
    fig.savefig(FIG_DIR / "fig1_pipeline.png")
    plt.close(fig)
    print("Figure 1: Pipeline diagram saved.")


def fig2_evolution_waterfall():
    """Figure 2: Phase evolution waterfall chart — Transfer Avg over time."""
    phases = [
        ("C0: BGE-small\nspecialist", 0.3011, C_SPECIALIST),
        ("C1: + Fixed A3\n(5% rescue)", 0.3028, C_CPU),
        ("C2: + Adaptive\ncoverage", 0.3098, C_CPU),
        ("C3: + B5 Batch\nMode-Switch", 0.3249, C_OURS),
        ("C4: + P3 Rank\nWindow Smoothing", 0.3293, C_BASELINE),
    ]
    bl_base = 0.3251  # BGE-base baseline
    bl_small = 0.3011  # BGE-small baseline

    fig, ax = plt.subplots(figsize=(9, 5))

    labels = [p[0] for p in phases]
    values = [p[1] for p in phases]
    colors = [p[2] for p in phases]
    deltas = [values[i] - values[i-1] if i > 0 else 0 for i in range(len(values))]

    # Waterfall bars
    x = np.arange(len(phases))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

    # Add delta labels
    for i in range(1, len(values)):
        ax.text(i, values[i] + 0.0025, f"+{deltas[i]:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_OURS)

    # Baseline lines
    ax.axhline(y=bl_base, color=C_GENERALIST, linestyle="--", linewidth=1.5, label=f"BGE-base generalist = {bl_base:.4f}")
    ax.axhline(y=bl_small, color=C_SPECIALIST, linestyle=":", linewidth=1.5, label=f"BGE-small specialist = {bl_small:.4f}")

    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.006,
                f"{val:.4f}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Transfer Average nDCG@10", fontsize=11)
    ax.set_title("SGAF Phase Evolution: Transfer Average Recovery", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.29, 0.34)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_evolution.pdf")
    fig.savefig(FIG_DIR / "fig2_evolution.png")
    plt.close(fig)
    print("Figure 2: Evolution waterfall saved.")


def fig3_per_dataset_bar():
    """Figure 3: Per-dataset nDCG@10 bar chart comparison."""
    datasets = ["SciFact", "NFCorpus", "FiQA", "SciDocs"]
    methods = ["BGE-small\nspecialist", "BGE-base\ngeneralist", "Frozen B5\nSGAF", "Frozen P3\nSGAF"]
    colors_list = [C_SPECIALIST, C_GENERALIST, C_OURS, C_BASELINE]

    data = {
        "SciFact":  [0.8188, 0.7376, 0.8218, 0.8218],
        "NFCorpus": [0.3505, 0.3695, 0.3692, 0.3744],
        "FiQA":     [0.3635, 0.3909, 0.3909, 0.3960],
        "SciDocs":  [0.1893, 0.2147, 0.2147, 0.2173],
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
    fig.suptitle("Per-Dataset nDCG@10 Comparison", fontsize=14, fontweight="bold")

    for i, ds in enumerate(datasets):
        ax = axes[i]
        x = np.arange(len(methods))
        vals = data[ds]
        bars = ax.bar(x, vals, color=colors_list, edgecolor="black", linewidth=0.8, width=0.6)

        # Value labels
        for bar, val in zip(bars, vals):
            if val < 0.3:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, val - 0.04,
                        f"{val:.4f}", ha="center", va="top", fontsize=7, color="white", fontweight="bold")

        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=6.5, rotation=30, ha="right")
        if ds == "SciFact":
            ax.set_ylim(0.70, 0.85)
        elif ds in ("NFCorpus",):
            ax.set_ylim(0.33, 0.39)
        elif ds == "FiQA":
            ax.set_ylim(0.34, 0.42)
        else:
            ax.set_ylim(0.17, 0.23)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_per_dataset.pdf")
    fig.savefig(FIG_DIR / "fig3_per_dataset.png")
    plt.close(fig)
    print("Figure 3: Per-dataset bar chart saved.")


def fig4_cost_pareto():
    """Figure 4: Cost-performance Pareto plot — latency vs nDCG@10."""
    methods = [
        ("BM25\n(Lucene)", 0.2247, 2.0, C_BASELINE, "o"),
        ("BGE-small\nSpecialist", 0.3011, 2.3, C_SPECIALIST, "s"),
        ("BGE-base\nGeneralist", 0.3251, 2.3, C_GENERALIST, "D"),
        ("Hybrid\nRRF", 0.3316, 9.0, C_CPU, "P"),
        ("Always\nRerank", 0.3909, 39.0, C_GPU, "X"),
        ("B5/P3\nSGAF (Ours)", 0.3293, 2.6, C_OURS, "*"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, ndcg, latency, color, marker in methods:
        ax.scatter(latency, ndcg, c=color, s=180 if name.startswith("B5") else 120,
                  marker=marker, edgecolors="black", linewidth=1.2, zorder=3)
        offset_x = 0.8
        offset_y = 0.004
        if name.startswith("B5"):
            offset_x = -1.5
            offset_y = 0.005
        ax.annotate(name, (latency, ndcg), textcoords="offset points",
                   xytext=(12, 5), fontsize=8, fontweight="bold" if name.startswith("B5") else "normal")

    # Pareto frontier indicator
    ax.annotate("← cheaper, faster", xy=(2, 0.215), fontsize=8, color="gray", style="italic")
    ax.annotate("better →", xy=(35, 0.395), fontsize=8, color="gray", style="italic", ha="right")

    ax.set_xlabel("Latency (ms/query, log scale)", fontsize=11)
    ax.set_ylabel("Transfer Avg nDCG@10", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(1, 50)
    ax.set_ylim(0.20, 0.42)
    ax.set_title("Cost-Performance Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=C_OURS, label="B5/P3 SGAF (CPU-only routing)"),
        mpatches.Patch(color=C_SPECIALIST, label="Specialist / Generalist"),
        mpatches.Patch(color=C_GPU, label="GPU-intensive (CE rerank)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_cost_pareto.pdf")
    fig.savefig(FIG_DIR / "fig4_cost_pareto.png")
    plt.close(fig)
    print("Figure 4: Cost Pareto plot saved.")


def fig5_external_validation():
    """Figure 5: External validation — TREC-COVID and ArguAna delta plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("External Validation: Frozen B5/P3 on Held-Out BEIR Datasets", fontsize=13, fontweight="bold")

    # TREC-COVID
    ax = axes[0]
    methods_trec = ["BGE-base", "Frozen B5", "Frozen P3"]
    vals_trec = [0.6835, 0.6835, 0.6908]
    colors_trec = [C_GENERALIST, C_OURS, C_BASELINE]
    x = np.arange(len(methods_trec))
    bars = ax.bar(x, vals_trec, color=colors_trec, edgecolor="black", width=0.5)
    for bar, val in zip(bars, vals_trec):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_trec, fontsize=9)
    ax.set_title("TREC-COVID (50 queries)\nP3 vs B5: +0.0073, p=0.47 (n.s.)", fontsize=10)
    ax.set_ylabel("nDCG@10", fontsize=10)
    ax.set_ylim(0.66, 0.72)
    ax.grid(axis="y", alpha=0.3)

    # ArguAna
    ax = axes[1]
    methods_arg = ["BGE-base", "Frozen B5", "Frozen P3"]
    vals_arg = [0.4530, 0.4301, 0.4312]
    colors_arg = [C_GENERALIST, C_OURS, C_BASELINE]
    x = np.arange(len(methods_arg))
    bars = ax.bar(x, vals_arg, color=colors_arg, edgecolor="black", width=0.5)
    for bar, val in zip(bars, vals_arg):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods_arg, fontsize=9)
    ax.set_title("ArguAna (1406 queries)\nP3 vs B5: +0.0011, *p=0.02", fontsize=10)
    ax.set_ylim(0.41, 0.48)
    ax.grid(axis="y", alpha=0.3)

    # Add note about B5 < BGE-base on ArguAna
    ax.annotate("B5 ← shift detector\nfails on argument retrieval",
                xy=(1, 0.4301), fontsize=7, color=C_OURS, style="italic",
                ha="center", va="top", xytext=(1, 0.44),
                arrowprops=dict(arrowstyle="->", color=C_OURS, lw=0.8))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_external.pdf")
    fig.savefig(FIG_DIR / "fig5_external.png")
    plt.close(fig)
    print("Figure 5: External validation saved.")


def main():
    print("Generating paper figures...")
    fig1_pipeline_diagram()
    fig2_evolution_waterfall()
    fig3_per_dataset_bar()
    fig4_cost_pareto()
    fig5_external_validation()
    print(f"\nAll 5 figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
