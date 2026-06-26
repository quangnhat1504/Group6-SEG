from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.types import Qrels, Run


def split_query_ids(query_ids: list[str], calibration_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(query_ids)
    rng.shuffle(shuffled)
    split_at = max(1, min(len(shuffled) - 1, round(len(shuffled) * calibration_fraction)))
    return set(shuffled[:split_at]), set(shuffled[split_at:])


def filter_run(run: Run, query_ids: set[str]) -> Run:
    return {query_id: run.get(query_id, []) for query_id in query_ids}


def filter_qrels(qrels: Qrels, query_ids: set[str]) -> Qrels:
    return {query_id: labels for query_id, labels in qrels.items() if query_id in query_ids}


def mean_regret(selected: Run, oracle_runs: dict[str, Run], qrels: Qrels) -> float:
    selected_scores = per_query_ndcg(selected, qrels, 10)
    oracle_scores = {}
    for query_id in qrels:
        oracle_scores[query_id] = max(
            per_query_ndcg({query_id: run.get(query_id, [])}, {query_id: qrels[query_id]}, 10)[query_id]
            for run in oracle_runs.values()
        )
    regrets = [oracle_scores[query_id] - selected_scores.get(query_id, 0.0) for query_id in qrels]
    return sum(regrets) / len(regrets) if regrets else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--llm-run", default="runs/scifact/test_test_llm_router_predictions_2_calibrated_heldout_routed.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    qrels_all = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    query_ids = sorted(qrels_all)
    calibration_ids, eval_ids = split_query_ids(query_ids, args.calibration_fraction, args.seed)
    qrels = filter_qrels(qrels_all, eval_ids)

    base_runs = {
        "BM25": load_run(config.outputs.run_dir / f"{args.split}_bm25.csv"),
        "Dense/SciNCL": load_run(config.outputs.run_dir / f"{args.split}_dense.csv"),
        "Hybrid RRF": load_run(config.outputs.run_dir / f"{args.split}_hybrid.csv"),
    }
    all_route_runs = {
        "bm25": base_runs["BM25"],
        "dense": base_runs["Dense/SciNCL"],
        "hybrid": base_runs["Hybrid RRF"],
    }
    runs = {name: filter_run(run, eval_ids) for name, run in base_runs.items()}
    llm_run_path = Path(args.llm_run)
    if llm_run_path.exists():
        runs["Small LLM Calibrated Held-Out"] = load_run(llm_run_path)

    rows = []
    for name, run in runs.items():
        metrics = evaluate_run(run, qrels)
        rows.append(
            {
                "method": name,
                "matched_queries": len(qrels),
                **metrics,
                "mean_regret@10": mean_regret(run, all_route_runs, qrels),
            }
        )

    output = Path(args.output) if args.output else config.outputs.run_dir / f"{args.split}_heldout_subset_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
