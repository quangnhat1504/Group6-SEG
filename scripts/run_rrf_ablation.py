"""RRF parameter (k) ablation for the Hybrid retriever.

Re-fuses the already-computed BM25 and Dense runs over a grid of rrf_k values and
evaluates each fused run. This is pure re-fusion from existing CSVs (no retrieval
re-run, no models), so it is cheap and fully reproducible.

Outputs:
  - runs/<split>_rrf_ablation.csv
  - reports/tables/table_rrf_ablation.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.fusion import reciprocal_rank_fusion
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.metrics import evaluate_run

K_GRID = (1, 5, 10, 20, 40, 60, 80, 100, 200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    bm25 = load_run(run_dir / f"{args.split}_bm25.csv")
    dense = load_run(run_dir / f"{args.split}_dense.csv")

    rows = []
    for k in K_GRID:
        fused = reciprocal_rank_fusion([bm25, dense], k=k, top_k=config.retrieval.top_k)
        metrics = evaluate_run(fused, qrels)
        rows.append({"rrf_k": k, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"{args.split}_rrf_ablation.csv", index=False)

    best = df.loc[df["ndcg@10"].idxmax()]
    default = df[df["rrf_k"] == config.retrieval.rrf_k].iloc[0]

    md = [
        f"RRF k ablation on SciFact {args.split} (re-fusion of BM25 + Dense/SciNCL). "
        f"Default config k={config.retrieval.rrf_k}.",
        "",
        "| rrf_k | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mark = "  **(default)**" if row["rrf_k"] == config.retrieval.rrf_k else ""
        md.append(
            f"| {row['rrf_k']}{mark} | {row['ndcg@10']:.4f} | {row['recall@10']:.4f} | "
            f"{row['recall@100']:.4f} | {row['mrr@10']:.4f} |"
        )
    md.append("")
    md.append(
        f"Best nDCG@10 at k={int(best['rrf_k'])} ({best['ndcg@10']:.4f}); "
        f"default k={config.retrieval.rrf_k} gives {default['ndcg@10']:.4f} "
        f"(delta {default['ndcg@10'] - best['ndcg@10']:+.4f})."
    )
    (reports / "tables" / "table_rrf_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))


if __name__ == "__main__":
    main()
