from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from seg_retrieval.config import ensure_output_dirs, load_config
from seg_retrieval.fusion import reciprocal_rank_fusion
from seg_retrieval.io import load_documents, load_qrels, load_queries, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.retrievers import BM25Retriever, DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    documents = load_documents(config.dataset.data_dir / f"{args.split}_documents.jsonl")
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")

    bm25_run = BM25Retriever(documents).search(queries, top_k=config.retrieval.top_k)
    save_run(config.outputs.run_dir / f"{args.split}_bm25.csv", bm25_run)
    metrics = {"bm25": evaluate_run(bm25_run, qrels)}

    if not args.skip_dense:
        dense_run = DenseRetriever(
            documents,
            model_name=config.retrieval.dense_model,
            batch_size=config.retrieval.dense_batch_size,
        ).search(queries, top_k=config.retrieval.top_k)
        hybrid_run = reciprocal_rank_fusion([bm25_run, dense_run], k=config.retrieval.rrf_k, top_k=config.retrieval.top_k)
        save_run(config.outputs.run_dir / f"{args.split}_dense.csv", dense_run)
        save_run(config.outputs.run_dir / f"{args.split}_hybrid.csv", hybrid_run)
        metrics["dense"] = evaluate_run(dense_run, qrels)
        metrics["hybrid"] = evaluate_run(hybrid_run, qrels)

    metrics_path = config.outputs.run_dir / f"{args.split}_base_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
