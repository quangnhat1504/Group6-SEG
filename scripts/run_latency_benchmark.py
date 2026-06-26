"""Wall-clock latency benchmark for the base retrievers (Phase 2 cost, measured).

Replaces the abstract Phase-2 cost units with real per-query online latency measured
on the actual hardware, so the whole report uses one consistent cost axis (ms/query).

Offline one-time costs (BM25 index build, dense document encoding) are reported
separately from the per-query online path (the quantity comparable to the
cross-encoder's 27.3 ms/query). Each per-query path is timed one query at a time
(batch=1) to reflect online serving, after a warm-up.

Outputs:
  - runs/<split>_latency_benchmark.json
  - reports/tables/table_latency.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from seg_retrieval.config import load_config
from seg_retrieval.fusion import reciprocal_rank_fusion
from seg_retrieval.io import load_documents, load_queries
from seg_retrieval.retrievers import BM25Retriever, DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    documents = load_documents(config.dataset.data_dir / f"{args.split}_documents.jsonl")
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    top_k = config.retrieval.top_k

    result: dict = {"split": args.split, "n_queries": len(queries), "n_documents": len(documents)}

    # ---- BM25: offline build, then per-query online search ----
    t0 = time.perf_counter()
    bm25 = BM25Retriever(documents)
    result["bm25_index_build_s"] = time.perf_counter() - t0
    result["bm25_backend"] = "rank_bm25" if bm25.index is not None else "naive_overlap"

    for q in queries[: args.warmup]:
        bm25.search([q], top_k=top_k)
    bm25_times = []
    for q in queries:
        t = time.perf_counter()
        bm25.search([q], top_k=top_k)
        bm25_times.append((time.perf_counter() - t) * 1000.0)
    result["bm25_ms_per_query"] = float(np.mean(bm25_times))

    # ---- Dense: offline doc encoding, then per-query online encode+search ----
    try:
        t0 = time.perf_counter()
        dense = DenseRetriever(
            documents,
            model_name=config.retrieval.dense_model,
            batch_size=config.retrieval.dense_batch_size,
        )
        result["dense_doc_encode_s"] = time.perf_counter() - t0
        result["dense_device"] = str(getattr(dense.model, "device", "unknown"))

        for q in queries[: args.warmup]:
            dense.search([q], top_k=top_k)
        dense_times, hybrid_times = [], []
        for q in queries:
            t = time.perf_counter()
            dense.search([q], top_k=top_k)
            dense_times.append((time.perf_counter() - t) * 1000.0)

            t = time.perf_counter()
            b_run = bm25.search([q], top_k=top_k)
            d_run = dense.search([q], top_k=top_k)
            reciprocal_rank_fusion([b_run, d_run], k=config.retrieval.rrf_k, top_k=top_k)
            hybrid_times.append((time.perf_counter() - t) * 1000.0)
        result["dense_ms_per_query"] = float(np.mean(dense_times))
        result["hybrid_ms_per_query"] = float(np.mean(hybrid_times))
        result["dense_available"] = True
    except Exception as exc:  # offline / model load failure
        result["dense_available"] = False
        result["dense_error"] = str(exc)

    (run_dir / f"{args.split}_latency_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    md = [
        f"Measured per-query online latency on SciFact {args.split} "
        f"({result['n_documents']} docs, {result['n_queries']} queries, batch=1, "
        f"after {args.warmup} warm-up queries). Offline one-time costs listed separately.",
        "",
        "| Stage | ms/query (online) | Offline one-time |",
        "|---|---:|---|",
        f"| BM25 ({result['bm25_backend']}) | {result['bm25_ms_per_query']:.2f} | "
        f"index build {result['bm25_index_build_s']:.1f}s |",
    ]
    if result.get("dense_available"):
        md.append(
            f"| Dense / SciNCL | {result['dense_ms_per_query']:.2f} | "
            f"doc encode {result['dense_doc_encode_s']:.1f}s ({result['dense_device']}) |"
        )
        md.append(f"| Hybrid RRF (BM25+Dense+fuse) | {result['hybrid_ms_per_query']:.2f} | - |")
    else:
        md.append("| Dense / SciNCL | (model unavailable offline) | - |")
    md.append("| Cross-encoder rerank (Hybrid top-20, GPU) | 27.3 | - (from Always-Rerank) |")
    (reports / "tables" / "table_latency.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\n" + json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
