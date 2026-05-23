from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_documents, load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.rerank import CrossEncoderReranker
from seg_retrieval.router import RouterPrediction
from seg_retrieval.uncertainty import decide_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dry-run", action="store_true", help="Only compute which queries would rerank.")
    args = parser.parse_args()

    config = load_config(args.config)
    documents = {doc.doc_id: doc for doc in load_documents(config.dataset.data_dir / f"{args.split}_documents.jsonl")}
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    bm25_run = load_run(config.outputs.run_dir / f"{args.split}_bm25.csv")
    dense_run = load_run(config.outputs.run_dir / f"{args.split}_dense.csv")
    hybrid_run = load_run(config.outputs.run_dir / f"{args.split}_hybrid.csv")

    labels_df = pd.read_csv(config.outputs.run_dir / f"{args.split}_oracle_labels.csv")
    label_by_query = dict(zip(labels_df["query_id"].astype(str), labels_df["oracle_label"].astype(str)))
    runs = {"bm25": bm25_run, "dense": dense_run, "hybrid": hybrid_run}
    selected_run = {
        query.query_id: runs.get(label_by_query.get(query.query_id, "hybrid"), hybrid_run).get(query.query_id, [])
        for query in queries
    }

    reranker = None if args.dry_run else CrossEncoderReranker(config.rerank.model)
    decisions = []
    start = time.perf_counter()
    for query in queries:
        label = label_by_query.get(query.query_id, "hybrid")
        prediction = RouterPrediction(label=label, confidence=1.0, probabilities={label: 1.0})
        decision = decide_uncertainty(
            query_id=query.query_id,
            prediction=prediction,
            selected_run=selected_run,
            bm25_run=bm25_run,
            dense_run=dense_run,
            router_confidence_threshold=config.rerank.router_confidence_threshold,
            score_gap_threshold=config.rerank.score_gap_threshold,
            disagreement_threshold=config.rerank.disagreement_threshold,
        )
        decisions.append({"query_id": query.query_id, "rerank": decision.should_rerank, "reasons": ";".join(decision.reasons), **decision.signals})
        if decision.should_rerank and reranker is not None:
            selected_run[query.query_id] = reranker.rerank(query, documents, selected_run[query.query_id], top_k=config.rerank.top_k)

    elapsed = time.perf_counter() - start
    metrics = evaluate_run(selected_run, qrels)
    metrics["rerank_coverage"] = sum(row["rerank"] for row in decisions) / len(decisions) if decisions else 0.0
    metrics["latency_ms_per_query"] = (elapsed / len(queries)) * 1000 if queries else 0.0
    if args.dry_run:
        metrics["mode"] = "dry_run_no_cross_encoder"

    save_run(config.outputs.run_dir / f"{args.split}_selective_rerank.csv", selected_run)
    pd.DataFrame(decisions).to_csv(config.outputs.run_dir / f"{args.split}_uncertainty_decisions.csv", index=False)
    (config.outputs.run_dir / f"{args.split}_selective_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
