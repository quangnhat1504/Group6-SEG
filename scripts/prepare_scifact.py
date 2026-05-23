from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seg_retrieval.config import ensure_output_dirs, load_config
from seg_retrieval.datasets import download_scifact, load_beir_split
from seg_retrieval.io import save_documents, save_qrels, save_queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_output_dirs(config)
    data_path = download_scifact(config.dataset.data_dir)
    documents, queries, qrels = load_beir_split(data_path, split=args.split)

    save_documents(config.dataset.data_dir / f"{args.split}_documents.jsonl", documents)
    save_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl", queries)
    save_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv", qrels)
    print(f"Saved {len(documents)} documents, {len(queries)} queries, {len(qrels)} qrels groups.")


if __name__ == "__main__":
    main()
