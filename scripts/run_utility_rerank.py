"""Direction 3: downstream-utility reranking (Always-Utility-Rerank).

Reranks the top-k Hybrid candidates of each query by a small LLM's claim-verification
label distribution (see src/seg_retrieval/utility_rerank.py): a candidate that makes
the LLM commit to SUPPORT/REFUTE (low semantic entropy, high verification confidence)
is treated as more useful evidence and ranked higher. Evaluated with the same
relevance nDCG@10 used elsewhere -- no gold verification labels needed.

The LLM forward passes are the expensive part, so every (query, doc) label
distribution is cached to a JSONL; reruns are free and the selective/gated variant
(compare_utility_rerank.py) reuses the same cache.

Outputs (under runs/<dataset>/):
  - <split>_utility_cache.jsonl      per (query, doc) label distribution (cache)
  - <split>_utility_rerank.csv       the Always-Utility reranked run
  - <split>_utility_features.csv     per-query base/utility nDCG + entropy summaries
  - <split>_utility_rerank_metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_documents, load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.utility_rerank import (
    LABELS,
    UtilityReranker,
    semantic_entropy,
    utility_from_dist,
    verification_confidence,
)


def load_cache(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    cache: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[(str(row["query_id"]), str(row["doc_id"]))] = {l: float(row[l]) for l in LABELS}
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--utility-mode", default="confidence", choices=["confidence", "entropy"])
    parser.add_argument("--top-k", type=int, default=None, help="Candidates to rerank (default config.rerank.top_k).")
    parser.add_argument("--max-queries", type=int, default=None, help="Process only the first N queries (pilots).")
    parser.add_argument("--want-4bit", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    top_k = args.top_k if args.top_k is not None else config.rerank.top_k

    documents = {d.doc_id: d for d in load_documents(config.dataset.data_dir / f"{args.split}_documents.jsonl")}
    queries = {q.query_id: q for q in load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")}
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    hybrid = load_run(run_dir / f"{args.split}_hybrid.csv")

    # Deterministic query order; restrict to queries we have a base ranking + labels for.
    query_ids = [qid for qid in qrels if qid in hybrid and qid in queries]
    if args.max_queries is not None:
        query_ids = query_ids[: args.max_queries]

    cache_path = run_dir / f"{args.split}_utility_cache.jsonl"
    cache = load_cache(cache_path)
    n_cached_start = len(cache)
    print(f"Loaded {n_cached_start} cached (query, doc) distributions from {cache_path}")

    # Count pairs still needing the LLM so we can decide whether to load the model.
    pending = [
        (qid, doc_id)
        for qid in query_ids
        for doc_id, _ in hybrid[qid][:top_k]
        if doc_id in documents and (qid, doc_id) not in cache
    ]
    print(f"Queries: {len(query_ids)}  top_k: {top_k}  pairs to score: {len(pending)} "
          f"(cache hits: {sum(len(hybrid[q][:top_k]) for q in query_ids) - len(pending)})")

    reranker = None
    if pending:
        print(f"Loading {args.model_id} ...", flush=True)
        t0 = time.time()
        reranker = UtilityReranker(model_name=args.model_id, utility_mode=args.utility_mode,
                                   want_4bit=args.want_4bit)
        print(f"Loaded in {time.time() - t0:.1f}s on {reranker.device}", flush=True)

    # Score (with incremental cache append) and build the reranked run.
    utility_run: dict[str, list[tuple[str, float]]] = {}
    feature_rows = []
    t_start = time.time()
    scored_since_flush = 0
    with cache_path.open("a", encoding="utf-8") as cache_file:
        for i, qid in enumerate(query_ids, 1):
            hits = hybrid[qid]
            candidates = [(d, s) for d, s in hits[:top_k] if d in documents]
            dists: dict[str, dict[str, float]] = {}
            for doc_id, _ in candidates:
                key = (qid, doc_id)
                dist = cache.get(key)
                if dist is None:
                    dist = reranker.label_distribution(queries[qid].text, documents[doc_id].text)
                    cache[key] = dist
                    cache_file.write(json.dumps({"query_id": qid, "doc_id": doc_id, **dist}) + "\n")
                    scored_since_flush += 1
                    if scored_since_flush >= 50:
                        cache_file.flush()
                        scored_since_flush = 0
                dists[doc_id] = dist

            reranked = sorted(
                ((d, utility_from_dist(dists[d], args.utility_mode)) for d, _ in candidates),
                key=lambda x: x[1], reverse=True,
            )
            scored_ids = {d for d, _ in candidates}
            untouched = [h for h in hits if h[0] not in scored_ids]
            utility_run[qid] = reranked + untouched

            ents = [semantic_entropy(dists[d]) for d, _ in candidates]
            confs = [verification_confidence(dists[d]) for d, _ in candidates]
            feature_rows.append({
                "query_id": qid,
                "n_candidates": len(candidates),
                "mean_entropy": sum(ents) / len(ents) if ents else 0.0,
                "mean_confidence": sum(confs) / len(confs) if confs else 0.0,
            })
            if i % 25 == 0 or i == len(query_ids):
                rate = (time.time() - t_start) / i
                print(f"  {i}/{len(query_ids)} queries  ({rate:.2f}s/query)", flush=True)

    save_run(run_dir / f"{args.split}_utility_rerank.csv", utility_run)
    print(f"Wrote {run_dir / (args.split + '_utility_rerank.csv')}  "
          f"({len(cache) - n_cached_start} new distributions cached)")

    # Claim-only baseline distributions (no abstract) for the InfoGain signal.
    baseline_path = run_dir / f"{args.split}_utility_baseline.jsonl"
    have_baseline = set()
    if baseline_path.exists():
        have_baseline = {json.loads(l)["query_id"] for l in baseline_path.open(encoding="utf-8") if l.strip()}
    missing = [qid for qid in query_ids if qid not in have_baseline]
    if missing:
        if reranker is None:
            reranker = UtilityReranker(model_name=args.model_id, utility_mode=args.utility_mode,
                                       want_4bit=args.want_4bit)
        with baseline_path.open("a", encoding="utf-8") as bf:
            for qid in missing:
                dist = reranker.claim_only_distribution(queries[qid].text)
                bf.write(json.dumps({"query_id": qid, **dist}) + "\n")
        print(f"Wrote {len(missing)} claim-only baselines to {baseline_path}")

    # Per-query nDCG, restricted to the processed queries for a fair comparison.
    sub_qrels = {qid: qrels[qid] for qid in query_ids}
    base_ndcg = per_query_ndcg(hybrid, sub_qrels, 10)
    util_ndcg = per_query_ndcg(utility_run, sub_qrels, 10)
    feats = pd.DataFrame(feature_rows)
    feats["base_ndcg"] = feats["query_id"].map(base_ndcg)
    feats["utility_ndcg"] = feats["query_id"].map(util_ndcg)
    feats["utility_gain"] = feats["utility_ndcg"] - feats["base_ndcg"]
    feats.to_csv(run_dir / f"{args.split}_utility_features.csv", index=False)

    metrics = evaluate_run(utility_run, sub_qrels)
    metrics["mode"] = "always_utility_rerank"
    metrics["n_queries"] = len(query_ids)
    metrics["top_k"] = top_k
    metrics["utility_mode"] = args.utility_mode
    metrics["model_id"] = args.model_id
    base_metrics = evaluate_run({q: hybrid[q] for q in query_ids}, sub_qrels)
    metrics["base_ndcg@10"] = base_metrics["ndcg@10"]
    (run_dir / f"{args.split}_utility_rerank_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    print("=" * 70)
    print("ALWAYS-UTILITY-RERANK — SUMMARY")
    print("=" * 70)
    print(f"Queries: {len(query_ids)}  utility_mode: {args.utility_mode}")
    print(f"Hybrid (base) nDCG@10 : {base_metrics['ndcg@10']:.4f}")
    print(f"Utility-Rerank nDCG@10: {metrics['ndcg@10']:.4f}  "
          f"({metrics['ndcg@10'] - base_metrics['ndcg@10']:+.4f})")
    print(f"recall@10={metrics['recall@10']:.4f}  mrr@10={metrics['mrr@10']:.4f}")
    print(f"\nWrote: {run_dir / (args.split + '_utility_features.csv')}")
    print(f"       {run_dir / (args.split + '_utility_rerank_metrics.json')}")


if __name__ == "__main__":
    main()
