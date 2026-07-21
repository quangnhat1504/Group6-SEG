"""Ablate adaptive coverage control for SGAF.

The BGE-base rescue ranking model is frozen from SciFact trainfit. This script
changes only the coverage budget. Label-free controllers estimate the budget
from source-vs-target distribution shift; oracle sweep rows are diagnostics and
must not be used as deployable settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from evaluate_frozen_sgaf_transfer import DATASETS, load_query_texts_and_qrels, load_runs, load_train_state, select_routes
from run_significance_conformal import paired_bootstrap
from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from train_query_adaptive_fusion import BASELINE, COMPONENTS, build_matrix, feature_names, oracle_labels, oracle_run, route_run


def source_feature_stats(config_path: str) -> dict[str, dict[str, float]]:
    config = load_config(config_path)
    run_dir = config.outputs.run_dir
    runs = {
        "bm25": load_run(run_dir / "trainfit_bm25.csv"),
        "bge_small": load_run(run_dir / "trainfit_dense_bge_small_scifact_rrf.csv"),
        "bge_base": load_run(run_dir / "trainfit_dense_bge_base.csv"),
    }
    qrels = load_qrels(config.dataset.data_dir / "trainfit_qrels.csv")
    queries = {query.query_id: query.text for query in load_queries(config.dataset.data_dir / "trainfit_queries.jsonl")}
    query_ids = sorted(qrels)
    x_train = build_matrix(runs, queries, query_ids)
    names = feature_names()
    tracked = ["query_len", "bge_small_top", "bge_small_gap", "bge_small_std10", "bge_small_bge_base_overlap10"]
    stats = {}
    for name in tracked:
        values = x_train[:, names.index(name)]
        stats[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)) if float(np.std(values)) > 1e-12 else 1.0,
        }
    return stats


def target_feature_means(runs, query_texts, query_ids) -> dict[str, float]:
    x_eval = build_matrix(runs, query_texts, query_ids)
    names = feature_names()
    tracked = ["query_len", "bge_small_top", "bge_small_gap", "bge_small_std10", "bge_small_bge_base_overlap10"]
    return {name: float(np.mean(x_eval[:, names.index(name)])) for name in tracked}


def zscores(means: dict[str, float], stats: dict[str, dict[str, float]]) -> dict[str, float]:
    return {name: (means[name] - values["mean"]) / values["std"] for name, values in stats.items()}


def clamp(value: float, lo: float = 0.02, hi: float = 0.4) -> float:
    return float(max(lo, min(hi, value)))


def source_shift_coverage(z: dict[str, float]) -> float:
    """Increase coverage when target looks unlike SciFact or specialist is flat."""
    value = (
        0.05
        + 0.05 * max(0.0, -z["query_len"])
        + 0.05 * abs(z["query_len"])
        + 0.05 * max(0.0, -z["bge_small_top"])
        + 0.05 * max(0.0, -z["bge_small_gap"])
    )
    return clamp(value)


def uncertainty_shift_coverage(z: dict[str, float]) -> float:
    """Ablation: adapt only from specialist uncertainty, ignoring query-domain shift."""
    value = 0.05 + 0.08 * max(0.0, -z["bge_small_gap"]) + 0.04 * max(0.0, -z["bge_small_std10"])
    return clamp(value)


def conservative_shift_coverage(z: dict[str, float]) -> float:
    """Ablation: a lower-variance source-shift controller."""
    value = 0.05 + 0.03 * abs(z["query_len"]) + 0.04 * max(0.0, -z["bge_small_gap"])
    return clamp(value, hi=0.25)


def evaluate_budget(name: str, method: str, coverage: float, model, class_index: int, runs, query_texts, qrels, output_dir: Path, params: dict) -> tuple[dict, dict]:
    query_ids = sorted(qrels)
    routes = select_routes(
        model=model,
        class_index=class_index,
        runs=runs,
        query_texts=query_texts,
        query_ids=query_ids,
        coverage=coverage,
    )
    run = route_run(routes, runs, query_ids)
    filename = f"{name}_{method.lower().replace(' ', '_').replace('/', '_')}.csv"
    save_run(output_dir / filename, run)
    metrics = evaluate_run(run, qrels, include_map=True)
    row = {
        "dataset": name,
        "method": method,
        "coverage": coverage,
        "switch_rate": sum(route != BASELINE for route in routes.values()) / len(routes),
        **metrics,
        "params": json.dumps(params, sort_keys=True),
        "run_path": str(output_dir / filename),
    }
    return row, run


def oracle_best_coverage(name: str, model, class_index: int, runs, query_texts, qrels, output_dir: Path, coverages: list[float]) -> tuple[dict, dict]:
    best_row = None
    best_run = None
    baseline_ndcg = evaluate_run(runs[BASELINE], qrels)["ndcg@10"]
    for coverage in coverages:
        row, run = evaluate_budget(
            name,
            f"Oracle coverage sweep {coverage:.2f}",
            coverage,
            model,
            class_index,
            runs,
            query_texts,
            qrels,
            output_dir,
            {"diagnostic": True, "coverage": coverage},
        )
        row["delta_vs_bge_small"] = row["ndcg@10"] - baseline_ndcg
        if best_row is None or row["ndcg@10"] > best_row["ndcg@10"]:
            best_row = row
            best_run = run
    assert best_row is not None and best_run is not None
    best_row["method"] = "Oracle best coverage sweep"
    return best_row, best_run


def component_rows(name: str, runs, qrels) -> list[dict]:
    baseline_ndcg = evaluate_run(runs[BASELINE], qrels)["ndcg@10"]
    rows = []
    for component in COMPONENTS:
        metrics = evaluate_run(runs[component], qrels, include_map=True)
        rows.append(
            {
                "dataset": name,
                "method": f"Component {component}",
                "coverage": None,
                "switch_rate": None,
                **metrics,
                "delta_vs_bge_small": metrics["ndcg@10"] - baseline_ndcg,
                "params": json.dumps({}),
                "run_path": "",
            }
        )
    labels = oracle_labels({c: per_query_ndcg(runs[c], qrels, 10) for c in COMPONENTS}, sorted(qrels))
    oracle = oracle_run({c: per_query_ndcg(runs[c], qrels, 10) for c in COMPONENTS}, runs, sorted(qrels))
    metrics = evaluate_run(oracle, qrels, include_map=True)
    rows.append(
        {
            "dataset": name,
            "method": "Oracle component router",
            "coverage": None,
            "switch_rate": None,
            **metrics,
            "delta_vs_bge_small": metrics["ndcg@10"] - baseline_ndcg,
            "params": json.dumps({"oracle_distribution": {c: labels.count(c) for c in COMPONENTS}}, sort_keys=True),
            "run_path": "",
        }
    )
    return rows


def add_deltas(rows: list[dict]) -> None:
    baseline_by_dataset = {
        row["dataset"]: row["ndcg@10"]
        for row in rows
        if row["method"] == f"Component {BASELINE}"
    }
    fixed_by_dataset = {
        row["dataset"]: row["ndcg@10"]
        for row in rows
        if row["method"] == "Fixed A3 coverage 5%"
    }
    for row in rows:
        row["delta_vs_bge_small"] = row["ndcg@10"] - baseline_by_dataset[row["dataset"]]
        row["delta_vs_fixed_a3"] = row["ndcg@10"] - fixed_by_dataset.get(row["dataset"], row["ndcg@10"])


def significance_rows(rows: list[dict], output_dir: Path, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    sig = []
    for dataset, spec in DATASETS.items():
        _, qrels = load_query_texts_and_qrels(spec)
        baseline_run = load_run(Path(spec["run_dir"]) / spec["runs"]["bge_small"])
        fixed_run = load_run(output_dir / f"{dataset}_fixed_a3_coverage_5%.csv")
        candidate_paths = {
            "Fixed A3 vs BGE-small": (fixed_run, baseline_run),
            "Source-shift adaptive vs fixed A3": (
                load_run(output_dir / f"{dataset}_source-shift_adaptive_coverage.csv"),
                fixed_run,
            ),
            "Uncertainty-shift adaptive vs fixed A3": (
                load_run(output_dir / f"{dataset}_uncertainty-shift_adaptive_coverage.csv"),
                fixed_run,
            ),
            "Conservative-shift adaptive vs fixed A3": (
                load_run(output_dir / f"{dataset}_conservative-shift_adaptive_coverage.csv"),
                fixed_run,
            ),
        }
        for comparison, (system_run, base_run) in candidate_paths.items():
            system_scores = per_query_ndcg(system_run, qrels, 10)
            base_scores = per_query_ndcg(base_run, qrels, 10)
            query_ids = sorted(qrels)
            system_vals = np.asarray([system_scores.get(qid, 0.0) for qid in query_ids])
            base_vals = np.asarray([base_scores.get(qid, 0.0) for qid in query_ids])
            mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(system_vals, base_vals, rng, n_boot=n_boot)
            sig.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "mean_delta_ndcg@10": mean_diff,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_value": p_value,
                    "significant": ci_lo > 0 or ci_hi < 0,
                }
            )
    return sig


def write_markdown(path: Path, rows: list[dict], sig: list[dict]) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Adaptive Coverage SGAF Ablation",
        "",
        "The BGE-base rescue ranking model is frozen from SciFact `trainfit`; this phase changes only the coverage budget.",
        "",
        "| Dataset | Method | Coverage | nDCG@10 | Delta vs BGE-small | Delta vs Fixed A3 | Recall@100 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    keep = [
        f"Component {BASELINE}",
        "Component bge_base",
        "Fixed A3 coverage 5%",
        "Source-shift adaptive coverage",
        "Uncertainty-shift adaptive coverage",
        "Conservative-shift adaptive coverage",
        "Oracle best coverage sweep",
        "Oracle component router",
    ]
    for dataset in DATASETS:
        subset = df[(df["dataset"] == dataset) & (df["method"].isin(keep))].copy()
        subset["_order"] = subset["method"].map({method: i for i, method in enumerate(keep)})
        subset = subset.sort_values("_order")
        for _, row in subset.iterrows():
            coverage = "N/A" if pd.isna(row["coverage"]) else f"{row['coverage']:.3f}"
            lines.append(
                f"| {dataset} | {row['method']} | {coverage} | {row['ndcg@10']:.4f} | "
                f"{row['delta_vs_bge_small']:+.4f} | {row['delta_vs_fixed_a3']:+.4f} | {row['recall@100']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Significance",
            "",
            "| Dataset | Comparison | Mean delta | 95% CI | p-value | Significant |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sig:
        lines.append(
            f"| {row['dataset']} | {row['comparison']} | {row['mean_delta_ndcg@10']:+.6f} | "
            f"[{row['ci_lo']:+.6f}, {row['ci_hi']:+.6f}] | {row['p_value']:.4f} | "
            f"{'yes' if row['significant'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Source-shift coverage is a cheap label-free controller that increases BGE-base coverage when target query/run statistics drift from SciFact trainfit.",
            "- Oracle coverage sweep is diagnostic only; it shows how much budget adaptation could matter if coverage were chosen perfectly per target.",
            "- The key risk is preserving SciFact: overly aggressive coverage hurts SciFact because BGE-base is globally weaker there.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--output-dir", default="runs/fusion/adaptive_coverage_sgaf")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, class_index, train_diagnostics = load_train_state(args.config, 0.1)
    stats = source_feature_stats(args.config)
    coverage_grid = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for dataset, spec in DATASETS.items():
        query_texts, qrels = load_query_texts_and_qrels(spec)
        runs = load_runs(spec)
        query_ids = sorted(qrels)
        means = target_feature_means(runs, query_texts, query_ids)
        z = zscores(means, stats)
        coverages = {
            "Fixed A3 coverage 5%": 0.05,
            "Source-shift adaptive coverage": source_shift_coverage(z),
            "Uncertainty-shift adaptive coverage": uncertainty_shift_coverage(z),
            "Conservative-shift adaptive coverage": conservative_shift_coverage(z),
        }
        diagnostics[dataset] = {"feature_means": means, "z_scores": z, "selected_coverages": coverages}
        rows.extend(component_rows(dataset, runs, qrels))
        for method, coverage in coverages.items():
            row, _ = evaluate_budget(dataset, method, coverage, model, class_index, runs, query_texts, qrels, output_dir, {"coverage": coverage, "z_scores": z})
            rows.append(row)
        best_row, _ = oracle_best_coverage(dataset, model, class_index, runs, query_texts, qrels, output_dir, coverage_grid)
        rows.append(best_row)

    add_deltas(rows)
    sig = significance_rows(rows, output_dir, args.n_boot, args.seed)
    rows_path = output_dir / "adaptive_coverage_sgaf_summary.csv"
    sig_path = output_dir / "adaptive_coverage_sgaf_significance.csv"
    json_path = output_dir / "adaptive_coverage_sgaf_summary.json"
    md_path = Path("reports/tables/table_adaptive_coverage_sgaf.md")
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    pd.DataFrame(sig).to_csv(sig_path, index=False)
    payload = {
        "protocol": {
            "gate": "SciFact trainfit binary BGE-base rescue classifier",
            "c_value": 0.1,
            "coverage_selection": "label-free source shift controllers; oracle sweep is diagnostic",
        },
        "train_diagnostics": train_diagnostics,
        "source_feature_stats": stats,
        "dataset_diagnostics": diagnostics,
        "rows": rows,
        "significance": sig,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, rows, sig)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {rows_path}")
    print(f"Wrote {sig_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
