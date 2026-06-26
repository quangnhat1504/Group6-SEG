"""Rerank depth ablation: top-50 vs existing top-20.

Reranks the top-50 candidates from Hybrid RRF (k=60) for all 300 test queries,
measures wall-clock latency, and runs a paired bootstrap significance test
comparing top-50 vs top-20 on nDCG@10.

Outputs:
  - runs/scifact/test_always_rerank_depth50.csv
  - runs/scifact/test_depth_ablation_metrics.json
  - reports/tables/table_rerank_depth.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from seg_retrieval.config import load_config
from seg_retrieval.io import load_documents, load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.rerank import CrossEncoderReranker
from seg_retrieval.types import Query

N_BOOT = 10000
SEED = 42


def paired_bootstrap(
    sys_vals: np.ndarray, base_vals: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT
) -> tuple[float, float, float, float]:
    """Paired bootstrap significance test. Returns (mean_diff, ci_lo, ci_hi, p)."""
    diff = sys_vals - base_vals
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_le = float(np.mean(means <= 0.0))
    frac_ge = float(np.mean(means >= 0.0))
    p = 2.0 * min(frac_le, frac_ge)
    return float(diff.mean()), float(lo), float(hi), min(p, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank depth ablation (top-50)")
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    data_dir = config.dataset.data_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading hybrid run, queries, documents, qrels...")
    hybrid_run = load_run(run_dir / f"{args.split}_hybrid.csv")
    qrels = load_qrels(data_dir / f"{args.split}_qrels.csv")
    queries = load_queries(data_dir / f"{args.split}_queries.jsonl")
    documents = load_documents(data_dir / f"{args.split}_documents.jsonl")
    doc_map = {doc.doc_id: doc for doc in documents}
    query_map = {q.query_id: q for q in queries}

    # Initialize reranker
    print("Initializing CrossEncoderReranker...")
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Rerank top-50 for all queries, measuring wall-clock time
    print(f"Reranking top-50 for {len(queries)} queries...")
    reranked_run = {}
    total_time = 0.0
    for q in queries:
        hits = hybrid_run.get(q.query_id, [])
        t0 = time.perf_counter()
        reranked_hits = reranker.rerank(q, doc_map, hits, top_k=50)
        t1 = time.perf_counter()
        total_time += (t1 - t0)
        reranked_run[q.query_id] = reranked_hits

    n_queries = len(queries)
    latency_ms_per_query_50 = (total_time / n_queries) * 1000.0
    print(f"  Total rerank time: {total_time:.2f}s, {latency_ms_per_query_50:.2f} ms/query")

    # Save reranked run
    save_run(run_dir / f"{args.split}_always_rerank_depth50.csv", reranked_run)
    print(f"  Saved: {run_dir}/{args.split}_always_rerank_depth50.csv")

    # Evaluate top-50 run
    metrics_50 = evaluate_run(reranked_run, qrels)
    print(f"  Top-50 metrics: {metrics_50}")

    # Load existing top-20 rerank results and metrics
    top20_run = load_run(run_dir / f"{args.split}_always_rerank.csv")
    metrics_20_path = run_dir / f"{args.split}_always_rerank_metrics.json"
    if metrics_20_path.exists():
        with open(metrics_20_path, "r", encoding="utf-8") as f:
            metrics_20_json = json.load(f)
        latency_ms_per_query_20 = metrics_20_json.get("latency_ms_per_query", None)
        ndcg_20 = metrics_20_json.get("ndcg@10", None)
        recall_20 = metrics_20_json.get("recall@10", None)
        mrr_20 = metrics_20_json.get("mrr@10", None)
    else:
        # Compute from loading the existing run
        print("  No existing metrics JSON found; computing top-20 metrics from run...")
        eval_20 = evaluate_run(top20_run, qrels)
        ndcg_20 = eval_20["ndcg@10"]
        recall_20 = eval_20["recall@10"]
        mrr_20 = eval_20["mrr@10"]
        latency_ms_per_query_20 = None  # Unknown without timing data

    # Paired bootstrap: top-50 vs top-20 on nDCG@10
    print("Running paired bootstrap (top-50 vs top-20 on nDCG@10)...")
    shared_queries = [q for q in qrels if q in reranked_run and q in top20_run]
    pq_50 = per_query_ndcg(reranked_run, {q: qrels[q] for q in shared_queries}, 10)
    pq_20 = per_query_ndcg(top20_run, {q: qrels[q] for q in shared_queries}, 10)
    arr_50 = np.array([pq_50[q] for q in shared_queries])
    arr_20 = np.array([pq_20[q] for q in shared_queries])

    rng = np.random.default_rng(SEED)
    mean_diff, ci_lo, ci_hi, p_val = paired_bootstrap(arr_50, arr_20, rng)
    significant = ci_lo > 0 or ci_hi < 0
    sig_label = "Yes" if significant else "No"
    print(f"  Mean diff: {mean_diff:+.4f}, 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}], p={p_val:.4f}, sig={sig_label}")

    # Build comparison table
    ndcg_50 = metrics_50["ndcg@10"]
    recall_50 = metrics_50["recall@10"]
    mrr_50 = metrics_50["mrr@10"]

    latency_20_str = f"{latency_ms_per_query_20:.2f}" if latency_ms_per_query_20 is not None else "N/A"
    latency_50_str = f"{latency_ms_per_query_50:.2f}"

    md_lines = [
        "# Rerank Depth Ablation: Top-20 vs Top-50",
        "",
        f"Paired bootstrap significance test ({N_BOOT} resamples, seed={SEED}) on nDCG@10.",
        "",
        "| Metric | Top-20 | Top-50 | Δ | Significant |",
        "|--------|-------:|-------:|--:|:-----------:|",
        f"| nDCG@10 | {ndcg_20:.4f} | {ndcg_50:.4f} | {ndcg_50 - ndcg_20:+.4f} | {sig_label} |",
        f"| Recall@10 | {recall_20:.4f} | {recall_50:.4f} | {recall_50 - recall_20:+.4f} | — |",
        f"| MRR@10 | {mrr_20:.4f} | {mrr_50:.4f} | {mrr_50 - mrr_20:+.4f} | — |",
        f"| Latency (ms/query) | {latency_20_str} | {latency_50_str} | — | — |",
        "",
        f"**Bootstrap CI (nDCG@10 diff):** [{ci_lo:+.4f}, {ci_hi:+.4f}], p = {p_val:.4f}",
        "",
        "**Note:** Al-Joofi et al. (2025) uses top-100 depth. Direct comparison of absolute nDCG "
        "is not valid due to protocol differences (100 vs 300 queries, different base systems).",
        "",
    ]

    table_path = reports / "tables" / "table_rerank_depth.md"
    table_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Saved: {table_path}")

    # Save metrics JSON
    depth_metrics = {
        "depth": 50,
        "latency_ms_per_query": latency_ms_per_query_50,
        "ndcg@10": ndcg_50,
        "recall@10": recall_50,
        "mrr@10": mrr_50,
    }
    metrics_path = run_dir / f"{args.split}_depth_ablation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(depth_metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Print summary
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
