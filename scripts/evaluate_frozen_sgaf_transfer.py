"""Evaluate frozen A3 Specialist-Generalist Adaptive Fusion on held-out datasets.

The gate is frozen from the SciFact trainfit recipe:
  - binary BGE-base rescue classifier
  - C=0.1
  - fixed coverage=0.05

No target-dataset labels are used for selecting routes. Target qrels are used
only for evaluation and oracle diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.datasets import load_beir_split_direct
from seg_retrieval.io import load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.types import Qrels, Query, Run

from train_query_adaptive_fusion import (
    BASELINE,
    COMPONENTS,
    build_matrix,
    component_scores,
    oracle_labels,
    oracle_run,
    route_distribution,
    route_run,
    train_classifier,
)


DATASETS = {
    "scifact": {
        "data_dir": "data/scifact",
        "run_dir": "runs/scifact",
        "queries": "test_queries.jsonl",
        "qrels": "test_qrels.csv",
        "runs": {
            "bm25": "test_bm25.csv",
            "bge_small": "test_dense_bge_small_scifact_rrf.csv",
            "bge_base": "test_dense_bge_base.csv",
        },
    },
    "nfcorpus": {
        "data_dir": "data/nfcorpus",
        "run_dir": "runs/nfcorpus",
        "beir": True,
        "runs": {
            "bm25": "test_bm25.csv",
            "bge_small": "test_dense_bge-small-final.csv",
            "bge_base": "test_dense_bge_base.csv",
        },
    },
    "fiqa": {
        "data_dir": "data/fiqa",
        "run_dir": "runs/fiqa",
        "queries": "test_queries.jsonl",
        "qrels": "test_qrels.csv",
        "runs": {
            "bm25": "test_bm25.csv",
            "bge_small": "test_dense_bge-small-final.csv",
            "bge_base": "test_dense_bge_base.csv",
        },
    },
    "scidocs": {
        "data_dir": "data/scidocs",
        "run_dir": "runs/scidocs",
        "beir": True,
        "runs": {
            "bm25": "test_bm25.csv",
            "bge_small": "test_dense_bge-small-final.csv",
            "bge_base": "test_dense_bge_base.csv",
        },
    },
}


def load_query_texts_and_qrels(spec: dict) -> tuple[dict[str, str], Qrels]:
    data_dir = Path(spec["data_dir"])
    if spec.get("beir"):
        _, queries, qrels = load_beir_split_direct(data_dir, "test")
        return {query.query_id: query.text for query in queries}, qrels
    queries = load_queries(data_dir / spec["queries"])
    qrels = load_qrels(data_dir / spec["qrels"])
    return {query.query_id: query.text for query in queries}, qrels


def load_runs(spec: dict) -> dict[str, Run]:
    run_dir = Path(spec["run_dir"])
    runs = {}
    for name, filename in spec["runs"].items():
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} run: {path}")
        runs[name] = load_run(path)
    return runs


def load_train_state(config_path: str, c_value: float):
    config = load_config(config_path)
    run_dir = config.outputs.run_dir
    paths = {
        "bm25": run_dir / "trainfit_bm25.csv",
        "bge_small": run_dir / "trainfit_dense_bge_small_scifact_rrf.csv",
        "bge_base": run_dir / "trainfit_dense_bge_base.csv",
    }
    train_runs = {name: load_run(path) for name, path in paths.items()}
    train_qrels = load_qrels(config.dataset.data_dir / "trainfit_qrels.csv")
    train_queries: list[Query] = load_queries(config.dataset.data_dir / "trainfit_queries.jsonl")
    train_texts = {query.query_id: query.text for query in train_queries}
    train_ids = sorted(train_qrels)
    train_scores = component_scores(train_runs, train_qrels)
    labels = [
        "bge_base"
        if train_scores["bge_base"].get(query_id, 0.0) > train_scores[BASELINE].get(query_id, 0.0)
        else BASELINE
        for query_id in train_ids
    ]
    x_train = build_matrix(train_runs, train_texts, train_ids)
    model = train_classifier(x_train, labels, c_value=c_value)
    if model is None:
        raise RuntimeError("Frozen A3 classifier could not be trained")
    classes = list(model.named_steps["clf"].classes_)
    if "bge_base" not in classes:
        raise RuntimeError("Frozen A3 classifier has no bge_base class")
    return model, classes.index("bge_base"), {
        "train_label_distribution": {
            "bge_base_rescue": labels.count("bge_base"),
            "specialist_default": labels.count(BASELINE),
        },
        "train_ids": len(train_ids),
    }


def select_routes(
    *,
    model,
    class_index: int,
    runs: dict[str, Run],
    query_texts: dict[str, str],
    query_ids: list[str],
    coverage: float,
) -> dict[str, str]:
    x_eval = build_matrix(runs, query_texts, query_ids)
    probs = model.predict_proba(x_eval)[:, class_index]
    n_selected = max(0, round(len(query_ids) * coverage))
    selected = {query_ids[index] for index in np.argsort(probs)[::-1][:n_selected]}
    return {query_id: "bge_base" if query_id in selected else BASELINE for query_id in query_ids}


def evaluate_dataset(name: str, spec: dict, model, class_index: int, coverage: float, output_dir: Path) -> list[dict]:
    query_texts, qrels = load_query_texts_and_qrels(spec)
    runs = load_runs(spec)
    query_ids = sorted(qrels)
    baseline_ndcg = evaluate_run(runs[BASELINE], qrels)["ndcg@10"]
    rows = []
    for component in COMPONENTS:
        metrics = evaluate_run(runs[component], qrels, include_map=True)
        rows.append(
            {
                "dataset": name,
                "stage": f"C:{component}",
                "method": f"Component {component}",
                "selected_on": "fixed",
                "switch_rate": None,
                **metrics,
                "delta_vs_bge_small": metrics["ndcg@10"] - baseline_ndcg,
                "params": json.dumps({}),
            }
        )

    scores = component_scores(runs, qrels)
    labels = oracle_labels(scores, query_ids)
    oracle = oracle_run(scores, runs, query_ids)
    oracle_metrics = evaluate_run(oracle, qrels, include_map=True)
    rows.append(
        {
            "dataset": name,
            "stage": "O1",
            "method": "Oracle component router",
            "selected_on": "oracle",
            "switch_rate": None,
            **oracle_metrics,
            "delta_vs_bge_small": oracle_metrics["ndcg@10"] - baseline_ndcg,
            "params": json.dumps({"oracle_distribution": {c: labels.count(c) for c in COMPONENTS}}, sort_keys=True),
        }
    )

    routes = select_routes(
        model=model,
        class_index=class_index,
        runs=runs,
        query_texts=query_texts,
        query_ids=query_ids,
        coverage=coverage,
    )
    frozen_run = route_run(routes, runs, query_ids)
    save_run(output_dir / f"{name}_frozen_a3_sgaf.csv", frozen_run)
    frozen_metrics = evaluate_run(frozen_run, qrels, include_map=True)
    rows.append(
        {
            "dataset": name,
            "stage": "A3-frozen",
            "method": "Frozen SciFact A3 SGAF",
            "selected_on": "scifact_trainfit",
            "switch_rate": sum(route != BASELINE for route in routes.values()) / len(routes),
            **frozen_metrics,
            "delta_vs_bge_small": frozen_metrics["ndcg@10"] - baseline_ndcg,
            "params": json.dumps(
                {"coverage": coverage, "routes": route_distribution(routes)},
                sort_keys=True,
            ),
        }
    )
    return rows


def write_markdown(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Frozen A3 SGAF Transfer",
        "",
        "Protocol: train the A3 gate on SciFact `trainfit`, freeze `C=0.1` and `coverage=0.05`, then apply the same gate to each target dataset without target-label tuning.",
        "",
        "| Dataset | Stage | Method | Switch | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["scifact", "nfcorpus", "fiqa", "scidocs"]
    stage_order = ["C:bm25", "C:bge_small", "C:bge_base", "O1", "A3-frozen"]
    for dataset in order:
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            continue
        subset["_stage_order"] = subset["stage"].map({stage: i for i, stage in enumerate(stage_order)})
        subset = subset.sort_values("_stage_order")
        for _, row in subset.iterrows():
            switch = "N/A" if pd.isna(row["switch_rate"]) else f"{row['switch_rate']:.4f}"
            lines.append(
                f"| {row['dataset']} | {row['stage']} | {row['method']} | {switch} | "
                f"{row['ndcg@10']:.4f} | {row['delta_vs_bge_small']:+.4f} | "
                f"{row['recall@10']:.4f} | {row['recall@100']:.4f} | {row['mrr@10']:.4f} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--output-dir", default="runs/fusion/frozen_sgaf_transfer")
    parser.add_argument("--coverage", type=float, default=0.05)
    parser.add_argument("--c-value", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, class_index, diagnostics = load_train_state(args.config, args.c_value)
    rows: list[dict] = []
    for name in args.datasets:
        rows.extend(evaluate_dataset(name, DATASETS[name], model, class_index, args.coverage, output_dir))
    rows_path = output_dir / "frozen_sgaf_transfer_summary.csv"
    json_path = output_dir / "frozen_sgaf_transfer_summary.json"
    md_path = Path("reports/tables/table_frozen_sgaf_transfer.md")
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    payload = {
        "protocol": {"c_value": args.c_value, "coverage": args.coverage, "selected_on": "scifact_trainfit"},
        "diagnostics": diagnostics,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, rows)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {rows_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
