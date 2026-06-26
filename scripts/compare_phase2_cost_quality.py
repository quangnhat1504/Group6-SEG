from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def method_cost(method: str) -> float:
    if method == "bm25":
        return 1.0
    if method == "dense":
        return 3.0
    if method == "hybrid":
        return 4.0
    return 0.0


def routed_cost(path: str | Path) -> tuple[float, dict[str, int]]:
    df = pd.read_csv(path, usecols=["query_id", "doc_id"])
    counts = {}
    for query_id, group in df.groupby("query_id"):
        # Routed CSVs do not preserve the route label. Infer from duplicated runs by row count elsewhere
        # is not reliable, so this helper is only used when a sibling predictions file is unavailable.
        counts[str(query_id)] = len(group)
    return 0.0, {"unknown": len(counts)}


def prediction_cost(path: str | Path) -> tuple[float, dict[str, int]]:
    df = pd.read_csv(path)
    if "pred_label" not in df.columns:
        raise ValueError(f"{path} must contain pred_label.")
    labels = df["pred_label"].astype(str).str.lower().value_counts().to_dict()
    total = sum(labels.values())
    avg_cost = sum(method_cost(label) * count for label, count in labels.items()) / max(total, 1)
    return avg_cost, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/scifact")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_metrics = read_json(run_dir / f"{args.split}_base_metrics.json")
    phase2_rows = pd.read_csv(run_dir / f"{args.split}_phase2_router_metrics.csv").to_dict("records")

    rows = []
    for method, metrics in base_metrics.items():
        label = method.lower()
        rows.append(
            {
                "category": "static",
                "method": method,
                "quality_policy": "single retrieval strategy",
                "estimated_cost_units_per_query": method_cost(label),
                "route_distribution": label,
                "matched_queries": 300,
                **metrics,
                "accuracy": None,
                "macro_f1": None,
            }
        )

    for row in phase2_rows:
        router = row["router"]
        if router == "Random Router":
            route_distribution = "expected uniform"
            cost = (method_cost("bm25") + method_cost("dense") + method_cost("hybrid")) / 3
        elif router == "Majority Router":
            route_distribution = "bm25=all"
            cost = method_cost("bm25")
        elif router == "Oracle Router":
            route_distribution = "oracle-dependent"
            cost = None
        else:
            route_distribution = "prediction-dependent"
            cost = None
        rows.append(
            {
                "category": "router",
                "method": router,
                "quality_policy": str(row.get("note", "")),
                "estimated_cost_units_per_query": cost,
                "route_distribution": route_distribution,
                "matched_queries": 300,
                "ndcg@10": row["ndcg@10"],
                "recall@10": row["recall@10"],
                "recall@100": row["recall@100"],
                "mrr@10": row["mrr@10"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
            }
        )

    llm_metric_files = sorted(run_dir.glob(f"{args.split}_*llm_router_predictions*metrics.json"))
    seen_llm_rows: set[tuple[str, float, float, float]] = set()
    for metric_file in llm_metric_files:
        if not metric_file.exists():
            continue
        metrics = read_json(metric_file)
        router_name = metrics["router"]
        dedupe_key = (
            router_name,
            round(float(metrics["ndcg@10"]), 12),
            round(float(metrics["accuracy"]), 12),
            float(metrics.get("matched_queries", 0)),
        )
        if dedupe_key in seen_llm_rows:
            continue
        seen_llm_rows.add(dedupe_key)
        prediction_file = Path(str(metric_file).replace("_metrics.json", "_predictions.csv"))
        if not prediction_file.exists() and "(2)" in metric_file.name:
            prediction_file = run_dir / "test_llm_router_predictions (2).csv"
        if not prediction_file.exists() and metric_file.name == f"{args.split}_test_llm_router_predictions_metrics.json":
            prediction_file = run_dir / "test_llm_router_predictions.csv"
        if not prediction_file.exists() and metric_file.name == f"{args.split}_test_llm_router_predictions_balanced_metrics.json":
            prediction_file = run_dir / "test_llm_router_predictions_balanced.csv"
        cost = None
        route_distribution = "unavailable"
        if prediction_file and prediction_file.exists():
            cost, labels = prediction_cost(prediction_file)
            route_distribution = ", ".join(f"{label}={count}" for label, count in sorted(labels.items()))
        rows.append(
            {
                "category": "llm-router",
                "method": router_name,
                "quality_policy": "LLM score route; calibrated row may use fallback",
                "estimated_cost_units_per_query": cost,
                "route_distribution": route_distribution,
                "matched_queries": metrics.get("matched_queries"),
                "ndcg@10": metrics["ndcg@10"],
                "recall@10": metrics["recall@10"],
                "recall@100": metrics["recall@100"],
                "mrr@10": metrics["mrr@10"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    output = run_dir / f"{args.split}_phase2_quality_cost.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
