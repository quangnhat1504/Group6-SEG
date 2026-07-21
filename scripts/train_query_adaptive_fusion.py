"""Train/evaluate query-adaptive specialist-generalist fusion.

The script learns only a lightweight query-level gate from cheap run-derived
features. It does not train retrievers. Current components are BM25,
BGE-small-final specialist, and BGE-base generalist.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.types import Qrels, Run


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
COMPONENTS = ["bm25", "bge_small", "bge_base"]
BASELINE = "bge_small"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def top_score(run: Run, query_id: str) -> float:
    hits = run.get(query_id, [])
    return float(hits[0][1]) if hits else 0.0


def score_gap(run: Run, query_id: str) -> float:
    hits = run.get(query_id, [])
    if not hits:
        return 0.0
    if len(hits) == 1:
        return float(hits[0][1])
    return float(hits[0][1] - hits[1][1])


def score_std(run: Run, query_id: str, depth: int = 10) -> float:
    values = [score for _, score in run.get(query_id, [])[:depth]]
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


def overlap_at_k(left: Run, right: Run, query_id: str, k: int) -> float:
    left_ids = {doc_id for doc_id, _ in left.get(query_id, [])[:k]}
    right_ids = {doc_id for doc_id, _ in right.get(query_id, [])[:k]}
    if not left_ids and not right_ids:
        return 0.0
    return len(left_ids & right_ids) / max(len(left_ids | right_ids), 1)


def rank_of_doc(run: Run, query_id: str, target_doc_id: str | None, default: int = 101) -> int:
    if target_doc_id is None:
        return default
    for rank, (doc_id, _) in enumerate(run.get(query_id, []), start=1):
        if doc_id == target_doc_id:
            return rank
    return default


def query_features(runs: dict[str, Run], query_text: str, query_id: str) -> list[float]:
    tokens = tokenize(query_text)
    features: list[float] = [
        float(len(tokens)),
        float(len(set(tokens))),
        float(sum(len(token) for token in tokens) / max(len(tokens), 1)),
    ]
    for name in COMPONENTS:
        run = runs[name]
        features.extend(
            [
                top_score(run, query_id),
                score_gap(run, query_id),
                score_std(run, query_id, 10),
            ]
        )
    for left, right in [("bm25", "bge_small"), ("bm25", "bge_base"), ("bge_small", "bge_base")]:
        features.extend(
            [
                overlap_at_k(runs[left], runs[right], query_id, 10),
                overlap_at_k(runs[left], runs[right], query_id, 20),
            ]
        )
        left_top = runs[left].get(query_id, [(None, 0.0)])[0][0]
        right_top = runs[right].get(query_id, [(None, 0.0)])[0][0]
        features.extend(
            [
                float(rank_of_doc(runs[right], query_id, left_top)),
                float(rank_of_doc(runs[left], query_id, right_top)),
            ]
        )
    return features


def feature_names() -> list[str]:
    names = ["query_len", "query_unique_terms", "query_avg_token_len"]
    for name in COMPONENTS:
        names.extend([f"{name}_top", f"{name}_gap", f"{name}_std10"])
    for left, right in [("bm25", "bge_small"), ("bm25", "bge_base"), ("bge_small", "bge_base")]:
        names.extend(
            [
                f"{left}_{right}_overlap10",
                f"{left}_{right}_overlap20",
                f"{left}_top_rank_in_{right}",
                f"{right}_top_rank_in_{left}",
            ]
        )
    return names


def build_matrix(runs: dict[str, Run], query_texts: dict[str, str], query_ids: list[str]) -> np.ndarray:
    return np.asarray([query_features(runs, query_texts.get(query_id, ""), query_id) for query_id in query_ids])


def component_scores(runs: dict[str, Run], qrels: Qrels) -> dict[str, dict[str, float]]:
    return {name: per_query_ndcg(run, qrels, 10) for name, run in runs.items()}


def oracle_labels(scores: dict[str, dict[str, float]], query_ids: list[str]) -> list[str]:
    labels = []
    tie_order = [BASELINE, "bge_base", "bm25"]
    for query_id in query_ids:
        best_score = max(scores[name].get(query_id, 0.0) for name in COMPONENTS)
        for name in tie_order:
            if math.isclose(scores[name].get(query_id, 0.0), best_score, abs_tol=1e-12):
                labels.append(name)
                break
    return labels


def route_run(routes: dict[str, str], runs: dict[str, Run], query_ids: list[str]) -> Run:
    output: Run = {}
    for query_id in query_ids:
        route = routes.get(query_id, BASELINE)
        output[query_id] = runs[route].get(query_id, [])
    return output


def oracle_run(scores: dict[str, dict[str, float]], runs: dict[str, Run], query_ids: list[str]) -> Run:
    labels = oracle_labels(scores, query_ids)
    return route_run(dict(zip(query_ids, labels)), runs, query_ids)


def train_classifier(x_train: np.ndarray, y_train: list[str], *, c_value: float) -> Pipeline | None:
    if len(set(y_train)) < 2:
        return None
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=13,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model


def evaluate_row(
    *,
    stage: str,
    method: str,
    run: Run,
    qrels: Qrels,
    baseline_ndcg: float,
    selected_on: str,
    params: dict,
    switch_rate: float | None = None,
) -> dict:
    metrics = evaluate_run(run, qrels, include_map=True)
    return {
        "stage": stage,
        "method": method,
        "selected_on": selected_on,
        "switch_rate": switch_rate,
        **metrics,
        "delta_vs_bge_small": metrics["ndcg@10"] - baseline_ndcg,
        "params": json.dumps(params, sort_keys=True),
    }


def route_distribution(routes: dict[str, str]) -> dict[str, int]:
    return {name: sum(1 for route in routes.values() if route == name) for name in COMPONENTS}


def zscore_by_train(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_z: dict[str, np.ndarray] = {}
    eval_z: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        values = x_train[:, index]
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std < 1e-12:
            std = 1.0
        train_z[name] = (x_train[:, index] - mean) / std
        eval_z[name] = (x_eval[:, index] - mean) / std
    return train_z, eval_z


def lexical_rescue_scores(z: dict[str, np.ndarray], formula: str) -> np.ndarray:
    if formula == "bm25_gap_minus_specialist_gap":
        return z["bm25_gap"] - z["bge_small_gap"]
    if formula == "bm25_top_minus_specialist_top":
        return z["bm25_top"] - z["bge_small_top"]
    if formula == "bm25_confident_low_overlap":
        return z["bm25_gap"] + z["bm25_std10"] - z["bm25_bge_small_overlap10"]
    if formula == "bm25_top_agrees_with_specialist_tail":
        return z["bm25_gap"] - z["bm25_top_rank_in_bge_small"]
    if formula == "bm25_disagrees_when_specialist_flat":
        return z["bm25_gap"] - z["bge_small_gap"] - z["bm25_bge_small_overlap20"]
    raise ValueError(f"Unknown lexical rescue formula: {formula}")


def coverage_selected(query_ids: list[str], scores: np.ndarray, coverage: float) -> set[str]:
    n_selected = max(0, round(len(query_ids) * coverage))
    if n_selected <= 0:
        return set()
    return {query_ids[index] for index in np.argsort(scores)[::-1][:n_selected]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--train-split", default="trainfit")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--output-dir", default="runs/fusion/query_adaptive_dev")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    paths = {
        "bm25": {
            "train": run_dir / f"{args.train_split}_bm25.csv",
            "eval": run_dir / f"{args.eval_split}_bm25.csv",
        },
        "bge_small": {
            "train": run_dir / f"{args.train_split}_dense_bge_small_scifact_rrf.csv",
            "eval": run_dir / f"{args.eval_split}_dense_bge_small_scifact_rrf.csv",
        },
        "bge_base": {
            "train": run_dir / f"{args.train_split}_dense_bge_base.csv",
            "eval": run_dir / f"{args.eval_split}_dense_bge_base.csv",
        },
    }
    for name, split_paths in paths.items():
        for split_name, path in split_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {split_name} run for {name}: {path}")

    train_runs = {name: load_run(spec["train"]) for name, spec in paths.items()}
    eval_runs = {name: load_run(spec["eval"]) for name, spec in paths.items()}
    train_qrels = load_qrels(config.dataset.data_dir / f"{args.train_split}_qrels.csv")
    eval_qrels = load_qrels(config.dataset.data_dir / f"{args.eval_split}_qrels.csv")
    train_queries = {q.query_id: q.text for q in load_queries(config.dataset.data_dir / f"{args.train_split}_queries.jsonl")}
    eval_queries = {q.query_id: q.text for q in load_queries(config.dataset.data_dir / f"{args.eval_split}_queries.jsonl")}
    train_ids = sorted(train_qrels)
    eval_ids = sorted(eval_qrels)

    train_scores = component_scores(train_runs, train_qrels)
    eval_scores = component_scores(eval_runs, eval_qrels)
    train_labels = oracle_labels(train_scores, train_ids)
    eval_oracle_labels = oracle_labels(eval_scores, eval_ids)
    x_train = build_matrix(train_runs, train_queries, train_ids)
    x_eval = build_matrix(eval_runs, eval_queries, eval_ids)

    baseline_metrics = evaluate_run(eval_runs[BASELINE], eval_qrels)
    baseline_ndcg = baseline_metrics["ndcg@10"]
    rows = []
    adaptive_outputs: dict[str, Run] = {}
    a3_train_routes: dict[str, str] | None = None
    a3_eval_routes: dict[str, str] | None = None
    for name in COMPONENTS:
        rows.append(
            evaluate_row(
                stage=f"C:{name}",
                method=f"Component {name}",
                run=eval_runs[name],
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on="fixed",
                params={},
            )
        )

    oracle_eval = oracle_run(eval_scores, eval_runs, eval_ids)
    rows.append(
        evaluate_row(
            stage="O1",
            method="Oracle component router",
            run=oracle_eval,
            qrels=eval_qrels,
            baseline_ndcg=baseline_ndcg,
            selected_on="oracle",
            params={"components": COMPONENTS},
        )
    )

    multiclass_candidates = []
    for c_value in [0.1, 1.0, 10.0]:
        model = train_classifier(x_train, train_labels, c_value=c_value)
        if model is None:
            continue
        pred_labels = list(model.predict(x_eval))
        routes = dict(zip(eval_ids, pred_labels))
        run = route_run(routes, eval_runs, eval_ids)
        multiclass_candidates.append(
            (
                evaluate_run(route_run(dict(zip(train_ids, model.predict(x_train))), train_runs, train_ids), train_qrels)[
                    "ndcg@10"
                ],
                c_value,
                routes,
                run,
            )
        )
    if multiclass_candidates:
        _, c_value, routes, run = max(multiclass_candidates, key=lambda item: item[0])
        adaptive_outputs["a1_multiclass_router"] = run
        rows.append(
            evaluate_row(
                stage="A1",
                method="Multiclass adaptive component router",
                run=run,
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=sum(route != BASELINE for route in routes.values()) / len(routes),
                params={"model": "logistic_regression", "c": c_value, "features": feature_names(), "routes": route_distribution(routes)},
            )
        )
    else:
        routes = {query_id: BASELINE for query_id in eval_ids}
        rows.append(
            evaluate_row(
                stage="A1",
                method="Multiclass adaptive component router no-op",
                run=eval_runs[BASELINE],
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=0.0,
                params={"reason": "single oracle class in train"},
            )
        )

    binary_labels = [
        "bge_base" if train_scores["bge_base"].get(query_id, 0.0) > train_scores[BASELINE].get(query_id, 0.0) else BASELINE
        for query_id in train_ids
    ]
    binary_candidates = []
    for c_value in [0.1, 1.0, 10.0]:
        model = train_classifier(x_train, binary_labels, c_value=c_value)
        if model is None:
            continue
        classes = list(model.named_steps["clf"].classes_)
        if "bge_base" not in classes:
            continue
        class_index = classes.index("bge_base")
        train_probs = model.predict_proba(x_train)[:, class_index]
        eval_probs = model.predict_proba(x_eval)[:, class_index]
        for threshold in [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]:
            train_routes = {
                query_id: "bge_base" if prob >= threshold else BASELINE
                for query_id, prob in zip(train_ids, train_probs)
            }
            eval_routes = {
                query_id: "bge_base" if prob >= threshold else BASELINE
                for query_id, prob in zip(eval_ids, eval_probs)
            }
            train_run = route_run(train_routes, train_runs, train_ids)
            eval_run = route_run(eval_routes, eval_runs, eval_ids)
            binary_candidates.append(
                (
                    evaluate_run(train_run, train_qrels)["ndcg@10"],
                    c_value,
                    threshold,
                    eval_routes,
                    eval_run,
                )
            )
    if binary_candidates:
        _, c_value, threshold, routes, run = max(binary_candidates, key=lambda item: item[0])
        adaptive_outputs["a2_bge_base_rescue_gate"] = run
        rows.append(
            evaluate_row(
                stage="A2",
                method="Binary BGE-base rescue gate",
                run=run,
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=sum(route != BASELINE for route in routes.values()) / len(routes),
                params={
                    "model": "logistic_regression",
                    "c": c_value,
                    "threshold": threshold,
                    "features": feature_names(),
                    "routes": route_distribution(routes),
                },
            )
        )
    else:
        rows.append(
            evaluate_row(
                stage="A2",
                method="Binary BGE-base rescue gate no-op",
                run=eval_runs[BASELINE],
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=0.0,
                params={"reason": "no bge_base rescue labels in train"},
            )
        )

    coverage_candidates = []
    for c_value in [0.1, 1.0, 10.0, 100.0]:
        model = train_classifier(x_train, binary_labels, c_value=c_value)
        if model is None:
            continue
        classes = list(model.named_steps["clf"].classes_)
        if "bge_base" not in classes:
            continue
        class_index = classes.index("bge_base")
        train_probs = model.predict_proba(x_train)[:, class_index]
        eval_probs = model.predict_proba(x_eval)[:, class_index]
        for coverage in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
            train_n = max(0, round(len(train_ids) * coverage))
            eval_n = max(0, round(len(eval_ids) * coverage))
            train_selected = {train_ids[index] for index in np.argsort(train_probs)[::-1][:train_n]}
            eval_selected = {eval_ids[index] for index in np.argsort(eval_probs)[::-1][:eval_n]}
            train_routes = {
                query_id: "bge_base" if query_id in train_selected else BASELINE
                for query_id in train_ids
            }
            eval_routes = {
                query_id: "bge_base" if query_id in eval_selected else BASELINE
                for query_id in eval_ids
            }
            train_run = route_run(train_routes, train_runs, train_ids)
            eval_run = route_run(eval_routes, eval_runs, eval_ids)
            coverage_candidates.append(
                (
                    evaluate_run(train_run, train_qrels)["ndcg@10"],
                    c_value,
                    coverage,
                    train_routes,
                    eval_routes,
                    eval_run,
                )
            )
    if coverage_candidates:
        _, c_value, coverage, train_routes, routes, run = max(coverage_candidates, key=lambda item: item[0])
        a3_train_routes = train_routes
        a3_eval_routes = routes
        adaptive_outputs["a3_coverage_controlled_bge_base_gate"] = run
        rows.append(
            evaluate_row(
                stage="A3",
                method="Coverage-controlled BGE-base rescue gate",
                run=run,
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=sum(route != BASELINE for route in routes.values()) / len(routes),
                params={
                    "model": "logistic_regression",
                    "c": c_value,
                    "coverage": coverage,
                    "features": feature_names(),
                    "routes": route_distribution(routes),
                },
            )
        )
    else:
        a3_train_routes = {query_id: BASELINE for query_id in train_ids}
        a3_eval_routes = {query_id: BASELINE for query_id in eval_ids}
        rows.append(
            evaluate_row(
                stage="A3",
                method="Coverage-controlled BGE-base rescue gate no-op",
                run=eval_runs[BASELINE],
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=0.0,
                params={"reason": "no bge_base rescue labels in train"},
            )
        )

    names = feature_names()
    train_z, eval_z = zscore_by_train(x_train, x_eval, names)
    bm25_formulas = [
        "bm25_gap_minus_specialist_gap",
        "bm25_top_minus_specialist_top",
        "bm25_confident_low_overlap",
        "bm25_top_agrees_with_specialist_tail",
        "bm25_disagrees_when_specialist_flat",
    ]
    bm25_coverages = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15]
    bm25_candidates = []
    for formula in bm25_formulas:
        train_signal = lexical_rescue_scores(train_z, formula)
        eval_signal = lexical_rescue_scores(eval_z, formula)
        for coverage in bm25_coverages:
            train_selected = coverage_selected(train_ids, train_signal, coverage)
            eval_selected = coverage_selected(eval_ids, eval_signal, coverage)
            train_routes = {query_id: "bm25" if query_id in train_selected else BASELINE for query_id in train_ids}
            eval_routes = {query_id: "bm25" if query_id in eval_selected else BASELINE for query_id in eval_ids}
            train_run = route_run(train_routes, train_runs, train_ids)
            eval_run = route_run(eval_routes, eval_runs, eval_ids)
            bm25_candidates.append(
                (
                    evaluate_run(train_run, train_qrels)["ndcg@10"],
                    formula,
                    coverage,
                    eval_routes,
                    eval_run,
                )
            )
    _, formula, coverage, routes, run = max(bm25_candidates, key=lambda item: item[0])
    adaptive_outputs["a4_coverage_controlled_bm25_gate"] = run
    rows.append(
        evaluate_row(
            stage="A4",
            method="Coverage-controlled BM25 lexical rescue gate",
            run=run,
            qrels=eval_qrels,
            baseline_ndcg=baseline_ndcg,
            selected_on=args.train_split,
            switch_rate=sum(route != BASELINE for route in routes.values()) / len(routes),
            params={"formula": formula, "coverage": coverage, "routes": route_distribution(routes)},
        )
    )

    dual_candidates = []
    if a3_train_routes is not None and a3_eval_routes is not None:
        for formula in bm25_formulas:
            train_signal = lexical_rescue_scores(train_z, formula)
            eval_signal = lexical_rescue_scores(eval_z, formula)
            for coverage in bm25_coverages:
                train_selected = coverage_selected(train_ids, train_signal, coverage)
                eval_selected = coverage_selected(eval_ids, eval_signal, coverage)
                train_routes = dict(a3_train_routes)
                eval_routes = dict(a3_eval_routes)
                for query_id in train_selected:
                    if train_routes.get(query_id, BASELINE) == BASELINE:
                        train_routes[query_id] = "bm25"
                for query_id in eval_selected:
                    if eval_routes.get(query_id, BASELINE) == BASELINE:
                        eval_routes[query_id] = "bm25"
                train_run = route_run(train_routes, train_runs, train_ids)
                eval_run = route_run(eval_routes, eval_runs, eval_ids)
                dual_candidates.append(
                    (
                        evaluate_run(train_run, train_qrels)["ndcg@10"],
                        formula,
                        coverage,
                        eval_routes,
                        eval_run,
                    )
                )
    if dual_candidates:
        _, formula, coverage, routes, run = max(dual_candidates, key=lambda item: item[0])
        adaptive_outputs["a5_dual_coverage_controlled_rescue_gate"] = run
        rows.append(
            evaluate_row(
                stage="A5",
                method="Dual coverage-controlled BGE-base + BM25 rescue gate",
                run=run,
                qrels=eval_qrels,
                baseline_ndcg=baseline_ndcg,
                selected_on=args.train_split,
                switch_rate=sum(route != BASELINE for route in routes.values()) / len(routes),
                params={"bm25_formula": formula, "bm25_coverage": coverage, "routes": route_distribution(routes)},
            )
        )

    diagnostics = {
        "train_label_distribution": {name: train_labels.count(name) for name in COMPONENTS},
        "eval_oracle_label_distribution": {name: eval_oracle_labels.count(name) for name in COMPONENTS},
        "feature_names": feature_names(),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / f"{args.train_split}_to_{args.eval_split}_query_adaptive_fusion.csv"
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    for name, run in adaptive_outputs.items():
        save_run(output_dir / f"{args.eval_split}_{name}.csv", run)
    best = max(rows, key=lambda row: (row["ndcg@10"], row["recall@10"], row["mrr@10"]))
    summary = {"rows_csv": str(rows_path), "best": best, "diagnostics": diagnostics, "rows": rows}
    summary_path = output_dir / f"{args.train_split}_to_{args.eval_split}_query_adaptive_fusion.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
