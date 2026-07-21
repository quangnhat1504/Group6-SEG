"""Run a saved SentenceTransformer dense model on a SciFact query split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from seg_retrieval.config import load_config
from seg_retrieval.io import load_documents, load_qrels, load_queries, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.retrievers import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--query-split", required=True)
    parser.add_argument("--documents-split", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--output-metrics", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.dataset.data_dir
    documents_split = args.documents_split or args.query_split
    documents_path = data_dir / f"{documents_split}_documents.jsonl"
    if not documents_path.exists():
        # SciFact local trainfit/dev are query subsets over the shared corpus.
        documents_path = data_dir / "test_documents.jsonl"
    queries_path = data_dir / f"{args.query_split}_queries.jsonl"
    qrels_path = data_dir / f"{args.query_split}_qrels.csv"

    documents = load_documents(documents_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    retriever = DenseRetriever(
        documents,
        model_name=args.model_path,
        batch_size=args.batch_size or config.retrieval.dense_batch_size,
    )
    run = retriever.search(queries, top_k=args.top_k or config.retrieval.top_k)
    output_run = Path(args.output_run)
    output_run.parent.mkdir(parents=True, exist_ok=True)
    save_run(output_run, run)

    metrics = {
        "query_split": args.query_split,
        "documents_path": str(documents_path),
        "model_path": args.model_path,
        "output_run": str(output_run),
        **evaluate_run(run, qrels),
    }
    if args.output_metrics:
        output_metrics = Path(args.output_metrics)
        output_metrics.parent.mkdir(parents=True, exist_ok=True)
        output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
