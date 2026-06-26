"""Direction 2: query-performance-prediction (QPP) signals for selective reranking.

Computes unsupervised QPP predictors on the BM25, Dense and Hybrid runs, then
correlates each with two targets:
  - base per-query nDCG@10 (how good is the un-reranked result), and
  - gain-from-reranking = Always-Rerank nDCG@10 - base nDCG@10 (the trigger target).

The predictor that best correlates with gain-from-reranking is the most useful
selective-reranking trigger and is fed to the conformal stage (run_conformal_rerank.py).

Outputs:
  - runs/<split>_qpp_features.csv          (per-query features + targets)
  - reports/tables/qpp_correlations.md/.csv
  - reports/figures/qpp_correlations.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats

from seg_retrieval.config import load_config
from seg_retrieval.fusion import top_doc_overlap
from seg_retrieval.io import load_qrels, load_queries, load_run
from seg_retrieval.metrics import per_query_ndcg
from seg_retrieval.qpp import corpus_mean, qpp_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10, help="Top-k depth for QPP statistics.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    bm25 = load_run(run_dir / f"{args.split}_bm25.csv")
    dense = load_run(run_dir / f"{args.split}_dense.csv")
    hybrid = load_run(run_dir / f"{args.split}_hybrid.csv")
    reranked = load_run(run_dir / f"{args.split}_always_rerank.csv")

    base_ndcg = per_query_ndcg(hybrid, qrels, 10)
    rerank_ndcg = per_query_ndcg(reranked, qrels, 10)

    mu = {"bm25": corpus_mean(bm25), "dense": corpus_mean(dense), "hybrid": corpus_mean(hybrid)}

    rows = []
    for q in queries:
        qid = q.query_id
        feats: dict[str, float] = {"query_id": qid}
        feats.update(qpp_features(bm25.get(qid, []), mu["bm25"], "bm25", args.k))
        feats.update(qpp_features(dense.get(qid, []), mu["dense"], "dense", args.k))
        feats.update(qpp_features(hybrid.get(qid, []), mu["hybrid"], "hybrid", args.k))
        # Cross-retriever (dis)agreement signals.
        overlap = top_doc_overlap(bm25.get(qid, []), dense.get(qid, []), k=10)
        feats["bm25_dense_overlap"] = overlap
        feats["disagreement"] = 1.0 - overlap
        # Targets.
        feats["base_ndcg"] = base_ndcg.get(qid, 0.0)
        feats["rerank_ndcg"] = rerank_ndcg.get(qid, 0.0)
        feats["gain"] = rerank_ndcg.get(qid, 0.0) - base_ndcg.get(qid, 0.0)
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"{args.split}_qpp_features.csv", index=False)

    feature_cols = [c for c in df.columns
                    if c not in ("query_id", "base_ndcg", "rerank_ndcg", "gain")]
    corr_rows = []
    for col in feature_cols:
        x = df[col].to_numpy()
        if np.std(x) < 1e-12:
            continue
        kt_base = stats.kendalltau(x, df["base_ndcg"]).statistic
        kt_gain = stats.kendalltau(x, df["gain"]).statistic
        pr_base = stats.pearsonr(x, df["base_ndcg"]).statistic
        pr_gain = stats.pearsonr(x, df["gain"]).statistic
        corr_rows.append({
            "feature": col,
            "kendall_vs_base_ndcg": kt_base,
            "kendall_vs_gain": kt_gain,
            "pearson_vs_base_ndcg": pr_base,
            "pearson_vs_gain": pr_gain,
            "abs_kendall_vs_gain": abs(kt_gain),
        })
    corr = pd.DataFrame(corr_rows).sort_values("abs_kendall_vs_gain", ascending=False)
    corr.to_csv(reports / "tables" / "qpp_correlations.csv", index=False)

    md = ["| Feature | Kendall vs base nDCG | Kendall vs gain | Pearson vs base nDCG | Pearson vs gain |",
          "|---|---:|---:|---:|---:|"]
    for r in corr.itertuples():
        md.append(f"| {r.feature} | {r.kendall_vs_base_ndcg:+.3f} | {r.kendall_vs_gain:+.3f} | "
                  f"{r.pearson_vs_base_ndcg:+.3f} | {r.pearson_vs_gain:+.3f} |")
    (reports / "tables" / "qpp_correlations.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Figure: |Kendall tau| vs each target.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = corr.sort_values("abs_kendall_vs_gain")
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(order))))
    ax.barh(y - 0.2, order["kendall_vs_gain"].abs(), height=0.4, label="|Kendall| vs gain-from-rerank", color="#1f77b4")
    ax.barh(y + 0.2, order["kendall_vs_base_ndcg"].abs(), height=0.4, label="|Kendall| vs base nDCG", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(order["feature"])
    ax.set_xlabel("|Kendall tau| (absolute correlation)")
    ax.set_title(f"QPP signal informativeness (SciFact {args.split})")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "qpp_correlations.png", dpi=150)

    best_gain = corr.iloc[0]
    best_base = corr.sort_values("kendall_vs_base_ndcg").iloc[0]  # most negative = low base -> rerank helps
    print("=" * 70)
    print("QPP SIGNALS — SUMMARY")
    print("=" * 70)
    print(f"Most informative for gain-from-reranking: {best_gain['feature']} "
          f"(Kendall {best_gain['kendall_vs_gain']:+.3f})")
    print(f"Most negative vs base nDCG (low base -> rerank helps): {best_base['feature']} "
          f"(Kendall {best_base['kendall_vs_base_ndcg']:+.3f})")
    print("\nTop features by |Kendall vs gain|:")
    print(corr[["feature", "kendall_vs_gain", "kendall_vs_base_ndcg"]].head(8).to_string(index=False))
    print(f"\nWrote: {run_dir / (args.split + '_qpp_features.csv')}")
    print(f"       {reports / 'tables' / 'qpp_correlations.md'}")
    print(f"       {reports / 'figures' / 'qpp_correlations.png'}")


if __name__ == "__main__":
    main()
