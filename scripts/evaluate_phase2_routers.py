from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.router import TfidfLogRegRouter, evaluate_router
from seg_retrieval.types import Query, Run


def route_run(query_ids: list[str], predictions: dict[str, str], runs: dict[str, Run]) -> Run:
    routed: Run = {}
    for query_id in query_ids:
        label = predictions[query_id]
        routed[query_id] = runs[label].get(query_id, [])
    return routed


def make_random_predictions(query_ids: list[str], labels: tuple[str, ...], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return {query_id: rng.choice(labels) for query_id in query_ids}


def make_majority_predictions(query_ids: list[str], oracle_labels: dict[str, str]) -> tuple[dict[str, str], str]:
    majority_label = Counter(oracle_labels.values()).most_common(1)[0][0]
    return {query_id: majority_label for query_id in query_ids}, majority_label


def make_classical_predictions(queries: list[Query], oracle_labels: dict[str, str], labels: tuple[str, ...]) -> dict[str, str]:
    router = TfidfLogRegRouter(labels=labels)
    y_train = [oracle_labels[query.query_id] for query in queries]
    router.fit(queries, y_train)
    return {query.query_id: prediction.label for query, prediction in zip(queries, router.predict(queries))}


def summarize_router(
    name: str,
    predictions: dict[str, str],
    oracle_labels: dict[str, str],
    query_ids: list[str],
    runs: dict[str, Run],
    qrels,
    note: str,
) -> tuple[dict, Run]:
    y_true = [oracle_labels[query_id] for query_id in query_ids]
    y_pred = [predictions[query_id] for query_id in query_ids]
    routed = route_run(query_ids, predictions, runs)
    routing_metrics = evaluate_router(y_true, y_pred)
    retrieval_metrics = evaluate_run(routed, qrels)
    row = {
        "router": name,
        **routing_metrics,
        **retrieval_metrics,
        "note": note,
    }
    return row, routed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = load_config(args.config)
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    labels_df = pd.read_csv(config.outputs.run_dir / f"{args.split}_oracle_labels.csv")
    oracle_labels = dict(zip(labels_df["query_id"].astype(str), labels_df["oracle_label"].astype(str)))
    query_ids = [query.query_id for query in queries if query.query_id in oracle_labels]

    runs = {
        "bm25": load_run(config.outputs.run_dir / f"{args.split}_bm25.csv"),
        "dense": load_run(config.outputs.run_dir / f"{args.split}_dense.csv"),
        "hybrid": load_run(config.outputs.run_dir / f"{args.split}_hybrid.csv"),
    }

    majority_predictions, majority_label = make_majority_predictions(query_ids, oracle_labels)
    prediction_sets = {
        "Random Router": (
            make_random_predictions(query_ids, config.router.label_order, args.seed),
            f"Uniform random over {', '.join(config.router.label_order)}; seed={args.seed}.",
        ),
        "Majority Router": (
            majority_predictions,
            f"Always routes to majority oracle class: {majority_label}.",
        ),
        "Oracle Router": (
            {query_id: oracle_labels[query_id] for query_id in query_ids},
            "Upper bound: uses oracle route labels directly.",
        ),
        "Classical TF-IDF LogReg Router": (
            make_classical_predictions(queries, oracle_labels, config.router.label_order),
            "Trained and evaluated on the same split; use as in-split sanity baseline.",
        ),
    }

    rows = []
    for name, (predictions, note) in prediction_sets.items():
        row, routed_run = summarize_router(name, predictions, oracle_labels, query_ids, runs, qrels, note)
        rows.append(row)
        safe_name = name.lower().replace(" ", "_").replace("-", "").replace("/", "_")
        save_run(config.outputs.run_dir / f"{args.split}_{safe_name}.csv", routed_run)

    output_json = config.outputs.run_dir / f"{args.split}_phase2_router_metrics.json"
    output_csv = config.outputs.run_dir / f"{args.split}_phase2_router_metrics.csv"
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
