from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_queries
from seg_retrieval.router import TfidfLogRegRouter, evaluate_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    query_map = {query.query_id: query for query in queries}
    labels_df = pd.read_csv(config.outputs.run_dir / f"{args.split}_oracle_labels.csv")
    train_queries = [query_map[str(query_id)] for query_id in labels_df["query_id"]]
    y_true = labels_df["oracle_label"].astype(str).tolist()

    router = TfidfLogRegRouter(labels=config.router.label_order)
    router.fit(train_queries, y_true)
    y_pred = [prediction.label for prediction in router.predict(train_queries)]
    metrics = evaluate_router(y_true, y_pred)
    output = config.outputs.run_dir / f"{args.split}_router_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
