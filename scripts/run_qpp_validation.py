"""Direction 2 Validation: QPP feature selection on the train split.

Demonstrates that hybrid_max was selected as the best predictor WITHOUT
data leakage — the same ranking of QPP signals holds on the train split
(809 queries) as on the test split (300 queries).

Steps:
  1. Load train-split BM25, Dense, Hybrid, Always-Rerank runs.
  2. Compute per-query nDCG@10 (base and reranked).
  3. Compute all QPP features per train query.
  4. Correlate each feature with base_ndcg and gain-from-reranking.
  5. Rank features by |Kendall tau vs gain| and check if hybrid_max is #1.
  6. Load test-split correlations and produce side-by-side comparison table.

Outputs:
  - runs/scifact/train_qpp_features.csv
  - reports/tables/qpp_train_correlations.csv
  - reports/tables/qpp_validation_comparison.md
"""
from __future__ import annotations

import argparse
import sys
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
    parser = argparse.ArgumentParser(
        description="Validate QPP feature selection on the train split (no data leakage)."
    )
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--k", type=int, default=10, help="Top-k depth for QPP statistics.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    # --- Check prerequisite files exist ---
    required_files = [
        run_dir / f"{args.split}_bm25.csv",
        run_dir / f"{args.split}_dense.csv",
        run_dir / f"{args.split}_hybrid.csv",
        run_dir / f"{args.split}_always_rerank.csv",
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("ERROR: Missing prerequisite run files:")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nTo generate train-split runs, execute:\n"
            "  python scripts/run_base_retrieval.py --split=train\n"
            "  python scripts/run_selective_rerank.py --split=train --rerank-all"
        )
        sys.exit(1)

    data_files = [
        config.dataset.data_dir / f"{args.split}_qrels.csv",
        config.dataset.data_dir / f"{args.split}_queries.jsonl",
    ]
    missing_data = [f for f in data_files if not Path(f).exists()]
    if missing_data:
        print("ERROR: Missing data files:")
        for f in missing_data:
            print(f"  - {f}")
        sys.exit(1)

    # --- Load data ---
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    bm25 = load_run(run_dir / f"{args.split}_bm25.csv")
    dense = load_run(run_dir / f"{args.split}_dense.csv")
    hybrid = load_run(run_dir / f"{args.split}_hybrid.csv")
    reranked = load_run(run_dir / f"{args.split}_always_rerank.csv")

    # --- Compute per-query nDCG targets ---
    base_ndcg = per_query_ndcg(hybrid, qrels, 10)
    rerank_ndcg = per_query_ndcg(reranked, qrels, 10)

    # --- Compute corpus means for QPP normalization ---
    mu = {
        "bm25": corpus_mean(bm25),
        "dense": corpus_mean(dense),
        "hybrid": corpus_mean(hybrid),
    }

    # --- Compute QPP features for each query ---
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
    print(f"Wrote: {run_dir / (args.split + '_qpp_features.csv')} ({len(df)} queries)")

    # --- Compute correlations ---
    feature_cols = [
        c for c in df.columns if c not in ("query_id", "base_ndcg", "rerank_ndcg", "gain")
    ]
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
    corr.to_csv(reports / "tables" / "qpp_train_correlations.csv", index=False)
    print(f"Wrote: {reports / 'tables' / 'qpp_train_correlations.csv'}")

    # --- Check if hybrid_max is the top predictor ---
    best_feature = corr.iloc[0]["feature"]
    best_kendall = corr.iloc[0]["kendall_vs_gain"]
    hybrid_max_row = corr[corr["feature"] == "hybrid_max"]

    print("\n" + "=" * 70)
    print("QPP VALIDATION - TRAIN SPLIT RESULTS")
    print("=" * 70)

    if best_feature == "hybrid_max":
        print(
            f"[OK] hybrid_max IS the top predictor on {args.split} split "
            f"(Kendall vs gain: {best_kendall:+.4f})"
        )
        print("  -> No data leakage concern: feature selection holds on train split.")
    else:
        print(
            f"[!!] hybrid_max is NOT the top predictor on {args.split} split."
        )
        print(f"  Top predictor: {best_feature} (Kendall vs gain: {best_kendall:+.4f})")
        if not hybrid_max_row.empty:
            hm_kendall = hybrid_max_row.iloc[0]["kendall_vs_gain"]
            hm_rank = corr.index.get_loc(hybrid_max_row.index[0]) + 1
            print(f"  hybrid_max: rank #{hm_rank} (Kendall vs gain: {hm_kendall:+.4f})")
        print("  -> Feature rankings differ between train and test; report both.")

    print("\nTop features by |Kendall vs gain|:")
    print(corr[["feature", "kendall_vs_gain", "kendall_vs_base_ndcg"]].head(8).to_string(index=False))

    # --- Load test-split correlations for side-by-side comparison ---
    test_corr_path = reports / "tables" / "qpp_correlations.csv"
    if not test_corr_path.exists():
        print(f"\nWARNING: Test-split correlations not found at {test_corr_path}")
        print("  Run `python scripts/run_qpp_signals.py --split=test` first.")
        print("  Skipping side-by-side comparison table.")
    else:
        test_corr = pd.read_csv(test_corr_path)

        # Merge train and test correlations
        train_subset = corr[["feature", "kendall_vs_gain", "pearson_vs_gain"]].rename(
            columns={
                "kendall_vs_gain": "train_kendall_vs_gain",
                "pearson_vs_gain": "train_pearson_vs_gain",
            }
        )
        test_subset = test_corr[["feature", "kendall_vs_gain", "pearson_vs_gain"]].rename(
            columns={
                "kendall_vs_gain": "test_kendall_vs_gain",
                "pearson_vs_gain": "test_pearson_vs_gain",
            }
        )
        comparison = train_subset.merge(test_subset, on="feature", how="outer").fillna(0.0)
        # Sort by train |Kendall| descending
        comparison["abs_train_kendall"] = comparison["train_kendall_vs_gain"].abs()
        comparison = comparison.sort_values("abs_train_kendall", ascending=False)

        # Identify top predictors for annotation
        train_top = comparison.iloc[0]["feature"]
        # Find test top by sorting on absolute test Kendall
        test_sorted = comparison.copy()
        test_sorted["abs_test_kendall"] = test_sorted["test_kendall_vs_gain"].abs()
        test_sorted = test_sorted.sort_values("abs_test_kendall", ascending=False)
        test_top = test_sorted.iloc[0]["feature"]

        # Build Markdown table
        md_lines = [
            "# QPP Validation: Train vs Test Split Correlations",
            "",
            f"Train split: {len(df)} queries | Test split: see `qpp_correlations.csv`",
            "",
            "| Feature | Train Kendall vs gain | Test Kendall vs gain | Note |",
            "|---|---:|---:|---|",
        ]
        for _, row in comparison.iterrows():
            feature = row["feature"]
            train_k = row["train_kendall_vs_gain"]
            test_k = row["test_kendall_vs_gain"]
            note = ""
            if feature == train_top and feature == test_top:
                note = "★ Top on both splits"
            elif feature == train_top:
                note = "★ Top on train"
            elif feature == test_top:
                note = "★ Top on test"
            md_lines.append(
                f"| {feature} | {train_k:+.4f} | {test_k:+.4f} | {note} |"
            )

        md_lines.extend([
            "",
            "## Summary",
            "",
            f"- **Train top predictor**: {train_top} "
            f"(|Kendall| = {comparison.iloc[0]['abs_train_kendall']:.4f})",
            f"- **Test top predictor**: {test_top} "
            f"(|Kendall| = {test_sorted.iloc[0]['abs_test_kendall']:.4f})",
        ])

        if train_top == test_top:
            md_lines.append(
                f"- **Conclusion**: {train_top} is the best predictor on BOTH splits "
                "→ no data leakage in feature selection."
            )
        else:
            md_lines.append(
                f"- **Note**: Top predictor differs between splits "
                f"(train: {train_top}, test: {test_top}). "
                "Both are reported for transparency."
            )

        md_content = "\n".join(md_lines) + "\n"
        comparison_path = reports / "tables" / "qpp_validation_comparison.md"
        comparison_path.write_text(md_content, encoding="utf-8")
        print(f"\nWrote: {comparison_path}")

        # Drop helper columns before finishing
        comparison.drop(columns=["abs_train_kendall"], inplace=True, errors="ignore")

    print(f"\nDone. All outputs in {run_dir}/ and {reports / 'tables'}/")


if __name__ == "__main__":
    main()
