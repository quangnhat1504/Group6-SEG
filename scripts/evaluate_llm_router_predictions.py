from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.router import evaluate_router
from seg_retrieval.types import Run


def normalize_label(value: str, allowed: tuple[str, ...]) -> str:
    text = str(value).strip().lower()
    for label in allowed:
        if text == label or text.startswith(label):
            return label
    if "bm25" in text:
        return "bm25"
    if "dense" in text:
        return "dense"
    if "hybrid" in text:
        return "hybrid"
    return "hybrid"


def route_run(query_ids: list[str], predictions: dict[str, str], runs: dict[str, Run]) -> Run:
    return {query_id: runs[predictions[query_id]].get(query_id, []) for query_id in query_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", required=True, help="CSV with query_id and pred_label columns.")
    parser.add_argument("--name", default="Small LLM QLoRA Router")
    args = parser.parse_args()

    config = load_config(args.config)
    predictions_df = pd.read_csv(args.predictions)
    if "query_id" not in predictions_df.columns:
        raise ValueError("Predictions file must contain a query_id column.")
    label_column = "pred_label" if "pred_label" in predictions_df.columns else "label"
    if label_column not in predictions_df.columns:
        raise ValueError("Predictions file must contain a pred_label or label column.")

    labels_df = pd.read_csv(config.outputs.run_dir / f"{args.split}_oracle_labels.csv")
    oracle_labels = dict(zip(labels_df["query_id"].astype(str), labels_df["oracle_label"].astype(str)))
    predictions = {
        str(row.query_id): normalize_label(getattr(row, label_column), config.router.label_order)
        for row in predictions_df.itertuples()
    }
    query_ids = [query_id for query_id in oracle_labels if query_id in predictions]
    missing_count = len(oracle_labels) - len(query_ids)
    if not query_ids:
        raise ValueError("No prediction query_id values matched oracle labels.")

    runs = {
        "bm25": load_run(config.outputs.run_dir / f"{args.split}_bm25.csv"),
        "dense": load_run(config.outputs.run_dir / f"{args.split}_dense.csv"),
        "hybrid": load_run(config.outputs.run_dir / f"{args.split}_hybrid.csv"),
    }
    routed = route_run(query_ids, predictions, runs)
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    y_true = [oracle_labels[query_id] for query_id in query_ids]
    y_pred = [predictions[query_id] for query_id in query_ids]
    metrics = {
        "router": args.name,
        "matched_queries": len(query_ids),
        "missing_predictions": missing_count,
        **evaluate_router(y_true, y_pred),
        **evaluate_run(routed, qrels),
    }

    stem = Path(args.predictions).stem
    save_run(config.outputs.run_dir / f"{args.split}_{stem}_routed.csv", routed)
    output = config.outputs.run_dir / f"{args.split}_{stem}_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
