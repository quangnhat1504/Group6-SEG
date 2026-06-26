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
    parser.add_argument("--train-split", default="train",
                        help="Split used to FIT the router.")
    parser.add_argument("--eval-split", default="test",
                        help="Disjoint split used to EVALUATE the router (no leakage).")
    args = parser.parse_args()

    config = load_config(args.config)

    def load_split(split: str):
        queries = load_queries(config.dataset.data_dir / f"{split}_queries.jsonl")
        query_map = {query.query_id: query for query in queries}
        labels_df = pd.read_csv(config.outputs.run_dir / f"{split}_oracle_labels.csv")
        ordered_queries = [query_map[str(query_id)] for query_id in labels_df["query_id"]]
        labels = labels_df["oracle_label"].astype(str).tolist()
        return ordered_queries, labels

    train_queries, y_train = load_split(args.train_split)
    eval_queries, y_eval = load_split(args.eval_split)

    router = TfidfLogRegRouter(labels=config.router.label_order)
    router.fit(train_queries, y_train)
    y_pred = [prediction.label for prediction in router.predict(eval_queries)]
    metrics = evaluate_router(y_eval, y_pred)
    output = config.outputs.run_dir / f"{args.eval_split}_router_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
