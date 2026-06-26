from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import random

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.router import evaluate_router
from seg_retrieval.types import Run


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def normalize_label(value: str, labels: tuple[str, ...]) -> str:
    text = str(value).strip().lower()
    for label in labels:
        if text == label or text.startswith(label):
            return label
    if "bm25" in text:
        return "bm25"
    if "dense" in text:
        return "dense"
    if "hybrid" in text:
        return "hybrid"
    return "hybrid"


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def predict_from_scores(
    df: pd.DataFrame,
    labels: tuple[str, ...],
    biases: dict[str, float],
    temperature: float,
    margin_threshold: float,
    fallback_label: str,
) -> pd.DataFrame:
    score_columns = [f"{label}_score" for label in labels]
    missing = [column for column in score_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    scores = df[score_columns].to_numpy(dtype=float)
    bias_vector = np.array([biases.get(label, 0.0) for label in labels], dtype=float)
    probs = softmax(scores + bias_vector, temperature)
    order = np.argsort(-probs, axis=1)
    top_idx = order[:, 0]
    second_idx = order[:, 1]
    top_probs = probs[np.arange(len(df)), top_idx]
    second_probs = probs[np.arange(len(df)), second_idx]
    margins = top_probs - second_probs

    calibrated = [labels[index] for index in top_idx]
    final = [
        label if margin >= margin_threshold else fallback_label
        for label, margin in zip(calibrated, margins)
    ]

    output = df.copy()
    output["raw_label"] = [
        normalize_label(value, labels)
        for value in output.get("pred_label", pd.Series(["hybrid"] * len(output)))
    ]
    output["calibrated_label"] = calibrated
    output["pred_label"] = final
    output["confidence"] = top_probs
    output["margin"] = margins
    output["fallback_used"] = margins < margin_threshold
    for index, label in enumerate(labels):
        output[f"calibrated_{label}_prob"] = probs[:, index]
    return output


def route_run(query_ids: list[str], predictions: dict[str, str], runs: dict[str, Run]) -> Run:
    return {query_id: runs[predictions[query_id]].get(query_id, []) for query_id in query_ids}


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    split: str,
    config,
    labels: tuple[str, ...],
) -> dict:
    labels_df = pd.read_csv(config.outputs.run_dir / f"{split}_oracle_labels.csv")
    oracle_labels = dict(zip(labels_df["query_id"].astype(str), labels_df["oracle_label"].astype(str)))
    predictions = {
        str(row.query_id): normalize_label(row.pred_label, labels)
        for row in predictions_df.itertuples()
    }
    query_ids = [query_id for query_id in oracle_labels if query_id in predictions]
    if not query_ids:
        raise ValueError("No prediction query_id values matched oracle labels.")

    runs = {
        "bm25": load_run(config.outputs.run_dir / f"{split}_bm25.csv"),
        "dense": load_run(config.outputs.run_dir / f"{split}_dense.csv"),
        "hybrid": load_run(config.outputs.run_dir / f"{split}_hybrid.csv"),
    }
    qrels_all = load_qrels(config.dataset.data_dir / f"{split}_qrels.csv")
    qrels = {query_id: qrels_all[query_id] for query_id in query_ids if query_id in qrels_all}
    routed = route_run(query_ids, predictions, runs)
    y_true = [oracle_labels[query_id] for query_id in query_ids]
    y_pred = [predictions[query_id] for query_id in query_ids]
    return {
        "matched_queries": len(query_ids),
        "missing_predictions": len(oracle_labels) - len(query_ids),
        "fallback_rate": float(predictions_df["fallback_used"].mean()),
        **evaluate_router(y_true, y_pred),
        **evaluate_run(routed, qrels),
    }, routed


def iter_biases(labels: tuple[str, ...], values: list[float]) -> list[dict[str, float]]:
    anchor = labels[-1]
    active_labels = labels[:-1]
    rows = []
    for combo in product(values, repeat=len(active_labels)):
        row = {label: value for label, value in zip(active_labels, combo)}
        row[anchor] = 0.0
        rows.append(row)
    return rows


def split_query_ids(df: pd.DataFrame, calibration_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1.")
    query_ids = sorted({str(query_id) for query_id in df["query_id"]})
    rng = random.Random(seed)
    rng.shuffle(query_ids)
    split_at = max(1, min(len(query_ids) - 1, round(len(query_ids) * calibration_fraction)))
    return set(query_ids[:split_at]), set(query_ids[split_at:])


def filter_queries(df: pd.DataFrame, query_ids: set[str]) -> pd.DataFrame:
    return df[df["query_id"].astype(str).isin(query_ids)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--calibration-split", default="test")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--eval-predictions", default=None)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=None,
        help="When calibration/eval use the same predictions file, split query IDs into calibration and held-out eval subsets.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--metric", choices=("ndcg@10", "macro_f1", "accuracy"), default="ndcg@10")
    parser.add_argument("--bias-min", type=float, default=-1.0)
    parser.add_argument("--bias-max", type=float, default=1.0)
    parser.add_argument("--bias-step", type=float, default=0.25)
    parser.add_argument("--temperatures", default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--margin-thresholds", default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--fallback-label", default="hybrid")
    parser.add_argument("--name", default="Small LLM QLoRA Router Calibrated")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = config.router.label_order
    if args.fallback_label not in labels:
        raise ValueError(f"--fallback-label must be one of {labels}")

    eval_split = args.eval_split or args.calibration_split
    eval_predictions_path = args.eval_predictions or args.calibration_predictions
    calibration_df = pd.read_csv(args.calibration_predictions)
    eval_df = pd.read_csv(eval_predictions_path)
    subset_note = None
    if args.calibration_fraction is not None:
        if str(Path(args.calibration_predictions).resolve()) != str(Path(eval_predictions_path).resolve()):
            raise ValueError("--calibration-fraction is only supported when calibration and eval use the same file.")
        calibration_ids, eval_ids = split_query_ids(calibration_df, args.calibration_fraction, args.seed)
        calibration_df = filter_queries(calibration_df, calibration_ids)
        eval_df = filter_queries(eval_df, eval_ids)
        subset_note = {
            "seed": args.seed,
            "calibration_fraction": args.calibration_fraction,
            "calibration_queries": len(calibration_ids),
            "eval_queries": len(eval_ids),
        }

    bias_values = np.arange(args.bias_min, args.bias_max + args.bias_step / 2, args.bias_step).round(6).tolist()
    temperatures = parse_float_list(args.temperatures)
    thresholds = parse_float_list(args.margin_thresholds)

    candidates = []
    for biases, temperature, threshold in product(iter_biases(labels, bias_values), temperatures, thresholds):
        calibrated = predict_from_scores(
            calibration_df,
            labels,
            biases=biases,
            temperature=temperature,
            margin_threshold=threshold,
            fallback_label=args.fallback_label,
        )
        metrics, _ = evaluate_predictions(calibrated, args.calibration_split, config, labels)
        candidates.append(
            {
                "temperature": temperature,
                "margin_threshold": threshold,
                **{f"{label}_bias": biases[label] for label in labels},
                **metrics,
            }
        )

    candidates_df = pd.DataFrame(candidates).sort_values(
        [args.metric, "macro_f1", "accuracy"],
        ascending=[False, False, False],
    )
    best = candidates_df.iloc[0].to_dict()
    best_biases = {label: float(best[f"{label}_bias"]) for label in labels}
    final_predictions = predict_from_scores(
        eval_df,
        labels,
        biases=best_biases,
        temperature=float(best["temperature"]),
        margin_threshold=float(best["margin_threshold"]),
        fallback_label=args.fallback_label,
    )
    final_metrics, routed = evaluate_predictions(final_predictions, eval_split, config, labels)
    final_metrics = {
        "router": args.name,
        "calibration_split": args.calibration_split,
        "eval_split": eval_split,
        "calibration_predictions": str(args.calibration_predictions),
        "eval_predictions": str(eval_predictions_path),
        "selected_metric": args.metric,
        "in_split_diagnostic": subset_note is None
        and str(Path(args.calibration_predictions).resolve())
        == str(Path(eval_predictions_path).resolve())
        and args.calibration_split == eval_split,
        "held_out_query_split": subset_note,
        "temperature": float(best["temperature"]),
        "margin_threshold": float(best["margin_threshold"]),
        "fallback_label": args.fallback_label,
        "biases": best_biases,
        **final_metrics,
    }

    stem = Path(eval_predictions_path).stem.replace(" ", "_").replace("(", "").replace(")", "")
    suffix = "calibrated_heldout" if subset_note else "calibrated"
    prefix = config.outputs.run_dir / f"{eval_split}_{stem}_{suffix}"
    final_predictions.to_csv(f"{prefix}_predictions.csv", index=False)
    save_run(f"{prefix}_routed.csv", routed)
    candidates_df.head(50).to_csv(f"{prefix}_tuning_top50.csv", index=False)
    Path(f"{prefix}_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
