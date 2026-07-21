"""Freeze and summarize the B5 SGAF mode-switch candidate.

Frozen candidate:
  - SciFact trainfit BGE-base rescue classifier, C=0.1
  - batch shift threshold = 2.0
  - shifted uncertainty gain = 6.0
  - shifted cap = 1.0

This script separates the frozen candidate from the exploratory B4/B5 sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from evaluate_frozen_sgaf_transfer import DATASETS, load_query_texts_and_qrels, load_runs, load_train_state, select_routes
from run_adaptive_coverage_sgaf import source_feature_stats, target_feature_means, zscores
from run_sgaf_mode_switch_ablation import batch_mode_coverage, batch_shift_score, uncertainty_coverage
from run_significance_conformal import paired_bootstrap
from seg_retrieval.io import load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from train_query_adaptive_fusion import BASELINE, route_run


METHOD_BGE_SMALL = "BGE-small specialist"
METHOD_BGE_BASE = "BGE-base generalist"
METHOD_CURRENT = "Current adaptive SGAF"
METHOD_B5 = "Frozen B5 mode-switch SGAF"


def leaked_scifact_test_ids(data_dir: Path) -> set[str]:
    train_queries = {query.query_id: query.text for query in load_queries(data_dir / "train_queries.jsonl")}
    test_queries = {query.query_id: query.text for query in load_queries(data_dir / "test_queries.jsonl")}
    train_qrels = load_qrels(data_dir / "train_qrels.csv")
    test_qrels = load_qrels(data_dir / "test_qrels.csv")
    train_text_to_ids: dict[str, list[str]] = {}
    for query_id, text in train_queries.items():
        train_text_to_ids.setdefault(text, []).append(query_id)

    leaked_ids: set[str] = set()
    for test_id, text in test_queries.items():
        test_gold = {doc_id for doc_id, rel in test_qrels.get(test_id, {}).items() if rel > 0}
        for train_id in train_text_to_ids.get(text, []):
            train_gold = {doc_id for doc_id, rel in train_qrels.get(train_id, {}).items() if rel > 0}
            if train_gold & test_gold:
                leaked_ids.add(test_id)
    return leaked_ids


def filter_run_qrels(run, qrels, exclude_query_ids: set[str]):
    return (
        {query_id: docs for query_id, docs in run.items() if query_id not in exclude_query_ids},
        {query_id: docs for query_id, docs in qrels.items() if query_id not in exclude_query_ids},
    )


def evaluate_named_run(dataset: str, method: str, run, qrels, output_dir: Path, params: dict | None = None, run_path: Path | None = None) -> dict:
    metrics = evaluate_run(run, qrels, include_map=True)
    return {
        "dataset": dataset,
        "method": method,
        **metrics,
        "params": json.dumps(params or {}, sort_keys=True),
        "run_path": str(run_path or ""),
    }


def evaluate_dataset(
    *,
    dataset: str,
    spec: dict,
    model,
    class_index: int,
    stats: dict[str, dict[str, float]],
    output_dir: Path,
    threshold: float,
    gain: float,
) -> tuple[list[dict], dict[str, dict], dict[str, object]]:
    query_texts, qrels = load_query_texts_and_qrels(spec)
    runs = load_runs(spec)
    query_ids = sorted(qrels)
    means = target_feature_means(runs, query_texts, query_ids)
    z = zscores(means, stats)

    current_coverage = uncertainty_coverage(z, gain=1.0, cap=0.4)
    b5_coverage, mode, shift_score = batch_mode_coverage(z, shift_threshold=threshold, shifted_gain=gain)

    current_routes = select_routes(
        model=model,
        class_index=class_index,
        runs=runs,
        query_texts=query_texts,
        query_ids=query_ids,
        coverage=current_coverage,
    )
    b5_routes = select_routes(
        model=model,
        class_index=class_index,
        runs=runs,
        query_texts=query_texts,
        query_ids=query_ids,
        coverage=b5_coverage,
    )

    current_run = route_run(current_routes, runs, query_ids)
    b5_run = route_run(b5_routes, runs, query_ids)
    current_path = output_dir / f"{dataset}_current_adaptive_sgaf.csv"
    b5_path = output_dir / f"{dataset}_frozen_b5_mode_switch_sgaf.csv"
    save_run(current_path, current_run)
    save_run(b5_path, b5_run)

    rows = [
        evaluate_named_run(dataset, METHOD_BGE_SMALL, runs[BASELINE], qrels, output_dir, run_path=Path(spec["run_dir"]) / spec["runs"][BASELINE]),
        evaluate_named_run(dataset, METHOD_BGE_BASE, runs["bge_base"], qrels, output_dir, run_path=Path(spec["run_dir"]) / spec["runs"]["bge_base"]),
        evaluate_named_run(
            dataset,
            METHOD_CURRENT,
            current_run,
            qrels,
            output_dir,
            params={"coverage": current_coverage, "mode": "current_uncertainty"},
            run_path=current_path,
        ),
        evaluate_named_run(
            dataset,
            METHOD_B5,
            b5_run,
            qrels,
            output_dir,
            params={
                "coverage": b5_coverage,
                "mode": mode,
                "batch_shift_score": shift_score,
                "threshold": threshold,
                "gain": gain,
            },
            run_path=b5_path,
        ),
    ]
    diagnostics = {
        "feature_means": means,
        "z_scores": z,
        "batch_shift_score": shift_score,
        "current_coverage": current_coverage,
        "b5_coverage": b5_coverage,
        "b5_mode": mode,
    }
    runs_for_sig = {
        METHOD_BGE_SMALL: runs[BASELINE],
        METHOD_BGE_BASE: runs["bge_base"],
        METHOD_CURRENT: current_run,
        METHOD_B5: b5_run,
    }
    return rows, runs_for_sig, diagnostics


def add_deltas(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    refs: dict[str, dict[str, float]] = {}
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        refs[dataset] = {
            "bge_small": float(subset[subset["method"] == METHOD_BGE_SMALL]["ndcg@10"].iloc[0]),
            "bge_base": float(subset[subset["method"] == METHOD_BGE_BASE]["ndcg@10"].iloc[0]),
            "current": float(subset[subset["method"] == METHOD_CURRENT]["ndcg@10"].iloc[0]),
        }
    for row in rows:
        ref = refs[row["dataset"]]
        row["delta_vs_bge_small"] = row["ndcg@10"] - ref["bge_small"]
        row["delta_vs_bge_base"] = row["ndcg@10"] - ref["bge_base"]
        row["delta_vs_current_adaptive"] = row["ndcg@10"] - ref["current"]


def method_summary(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    summary = []
    for method, subset in df.groupby("method", sort=False):
        transfer = subset[subset["dataset"] != "scifact"]
        summary.append(
            {
                "method": method,
                "avg_ndcg@10": float(subset["ndcg@10"].mean()),
                "transfer_avg_ndcg@10": float(transfer["ndcg@10"].mean()),
                "avg_delta_vs_bge_small": float(subset["delta_vs_bge_small"].mean()),
                "transfer_delta_vs_bge_base": float(transfer["delta_vs_bge_base"].mean()),
                "scifact_delta_vs_bge_small": float(subset[subset["dataset"] == "scifact"]["delta_vs_bge_small"].iloc[0]),
                "scifact_delta_vs_current": float(subset[subset["dataset"] == "scifact"]["delta_vs_current_adaptive"].iloc[0]),
            }
        )
    return summary


def duplicate_filtered_scifact(output_dir: Path) -> list[dict]:
    data_dir = Path("data/scifact")
    leaked_ids = leaked_scifact_test_ids(data_dir)
    qrels = load_qrels(data_dir / "test_qrels.csv")
    runs = {
        METHOD_BGE_SMALL: load_run(Path(DATASETS["scifact"]["run_dir"]) / DATASETS["scifact"]["runs"][BASELINE]),
        METHOD_BGE_BASE: load_run(Path(DATASETS["scifact"]["run_dir"]) / DATASETS["scifact"]["runs"]["bge_base"]),
        METHOD_CURRENT: load_run(output_dir / "scifact_current_adaptive_sgaf.csv"),
        METHOD_B5: load_run(output_dir / "scifact_frozen_b5_mode_switch_sgaf.csv"),
    }
    rows = []
    for method, run in runs.items():
        clean_run, clean_qrels = filter_run_qrels(run, qrels, leaked_ids)
        full = evaluate_run(run, qrels, include_map=True)
        filtered = evaluate_run(clean_run, clean_qrels, include_map=True)
        rows.append(
            {
                "method": method,
                "excluded_query_ids": ",".join(sorted(leaked_ids)),
                "full_ndcg@10": full["ndcg@10"],
                "filtered_ndcg@10": filtered["ndcg@10"],
                "full_recall@10": full["recall@10"],
                "filtered_recall@10": filtered["recall@10"],
                "full_recall@100": full["recall@100"],
                "filtered_recall@100": filtered["recall@100"],
            }
        )
    baseline = next(row["filtered_ndcg@10"] for row in rows if row["method"] == METHOD_BGE_SMALL)
    current = next(row["filtered_ndcg@10"] for row in rows if row["method"] == METHOD_CURRENT)
    for row in rows:
        row["filtered_delta_vs_bge_small"] = row["filtered_ndcg@10"] - baseline
        row["filtered_delta_vs_current_adaptive"] = row["filtered_ndcg@10"] - current
    return rows


def significance_rows(runs_by_dataset: dict[str, dict[str, object]], n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [METHOD_CURRENT, METHOD_BGE_SMALL, METHOD_BGE_BASE]
    for dataset, runs in runs_by_dataset.items():
        _, qrels = load_query_texts_and_qrels(DATASETS[dataset])
        system_scores = per_query_ndcg(runs[METHOD_B5], qrels, 10)
        query_ids = sorted(qrels)
        system_vals = np.asarray([system_scores.get(query_id, 0.0) for query_id in query_ids])
        for baseline in comparisons:
            baseline_scores = per_query_ndcg(runs[baseline], qrels, 10)
            baseline_vals = np.asarray([baseline_scores.get(query_id, 0.0) for query_id in query_ids])
            mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(system_vals, baseline_vals, rng, n_boot=n_boot)
            rows.append(
                {
                    "dataset": dataset,
                    "system": METHOD_B5,
                    "baseline": baseline,
                    "queries": len(query_ids),
                    "mean_delta_ndcg@10": mean_diff,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_value": p_value,
                    "significant": ci_lo > 0 or ci_hi < 0,
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict], summary: list[dict], dup_rows: list[dict], sig_rows: list[dict], diagnostics: dict) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Final B5 SGAF Mode-Switch Candidate",
        "",
        "Frozen recipe: SciFact trainfit BGE-base rescue classifier (`C=0.1`), batch shift threshold `2.0`, shifted uncertainty gain `6.0`, shifted cap `1.0`.",
        "",
        "## Summary",
        "",
        "| Method | Avg nDCG@10 | Transfer Avg | Avg delta vs BGE-small | Transfer delta vs BGE-base | SciFact delta vs BGE-small | SciFact delta vs current |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['avg_ndcg@10']:.4f} | {row['transfer_avg_ndcg@10']:.4f} | "
            f"{row['avg_delta_vs_bge_small']:+.4f} | {row['transfer_delta_vs_bge_base']:+.4f} | "
            f"{row['scifact_delta_vs_bge_small']:+.4f} | {row['scifact_delta_vs_current']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Detail",
            "",
            "| Dataset | Method | Coverage | Mode | Shift | nDCG@10 | Delta vs BGE-small | Delta vs BGE-base | Delta vs current | Recall@100 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    order = [METHOD_BGE_SMALL, METHOD_BGE_BASE, METHOD_CURRENT, METHOD_B5]
    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset].copy()
        subset["_order"] = subset["method"].map({method: index for index, method in enumerate(order)})
        subset = subset.sort_values("_order")
        diag = diagnostics[dataset]
        for _, row in subset.iterrows():
            params = json.loads(row["params"])
            coverage = params.get("coverage")
            coverage_text = "N/A" if coverage is None else f"{coverage:.3f}"
            mode = params.get("mode", "component")
            shift = diag["batch_shift_score"]
            lines.append(
                f"| {dataset} | {row['method']} | {coverage_text} | {mode} | {shift:.3f} | "
                f"{row['ndcg@10']:.4f} | {row['delta_vs_bge_small']:+.4f} | "
                f"{row['delta_vs_bge_base']:+.4f} | {row['delta_vs_current_adaptive']:+.4f} | "
                f"{row['recall@100']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Duplicate-Filtered SciFact",
            "",
            "| Method | Full nDCG@10 | Filtered nDCG@10 | Filtered delta vs BGE-small | Filtered delta vs current | Full Recall@10 | Filtered Recall@10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dup_rows:
        lines.append(
            f"| {row['method']} | {row['full_ndcg@10']:.4f} | {row['filtered_ndcg@10']:.4f} | "
            f"{row['filtered_delta_vs_bge_small']:+.4f} | {row['filtered_delta_vs_current_adaptive']:+.4f} | "
            f"{row['full_recall@10']:.4f} | {row['filtered_recall@10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Bootstrap",
            "",
            "| Dataset | Baseline | Mean delta | 95% CI | p-value | Significant |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sig_rows:
        lines.append(
            f"| {row['dataset']} | {row['baseline']} | {row['mean_delta_ndcg@10']:+.6f} | "
            f"[{row['ci_lo']:+.6f}, {row['ci_hi']:+.6f}] | {row['p_value']:.4f} | "
            f"{'yes' if row['significant'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Frozen B5 keeps the current SciFact adaptive score because SciFact is source-like under the batch shift score.",
            "- On shifted datasets, the controller increases BGE-base coverage and nearly matches the BGE-base transfer average.",
            "- This remains a Phase 7 candidate because threshold/gain were identified in the exploratory B4/B5 sweep; the next paper-grade step is freezing these values before any new dataset or held-out batch.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--output-dir", default="runs/fusion/final_sgaf_mode_switch")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--gain", type=float, default=6.0)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, class_index, train_diagnostics = load_train_state(args.config, 0.1)
    stats = source_feature_stats(args.config)
    all_rows: list[dict] = []
    runs_by_dataset: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, dict] = {}
    for dataset, spec in DATASETS.items():
        rows, runs_for_sig, diag = evaluate_dataset(
            dataset=dataset,
            spec=spec,
            model=model,
            class_index=class_index,
            stats=stats,
            output_dir=output_dir,
            threshold=args.threshold,
            gain=args.gain,
        )
        all_rows.extend(rows)
        runs_by_dataset[dataset] = runs_for_sig
        diagnostics[dataset] = diag

    add_deltas(all_rows)
    summary = method_summary(all_rows)
    dup_rows = duplicate_filtered_scifact(output_dir)
    sig_rows = significance_rows(runs_by_dataset, args.n_boot, args.seed)

    rows_path = output_dir / "final_sgaf_mode_switch_rows.csv"
    summary_path = output_dir / "final_sgaf_mode_switch_summary.csv"
    dup_path = output_dir / "final_sgaf_mode_switch_duplicate_filtered_scifact.csv"
    sig_path = output_dir / "final_sgaf_mode_switch_significance.csv"
    json_path = output_dir / "final_sgaf_mode_switch_manifest.json"
    md_path = Path("reports/tables/table_final_sgaf_mode_switch.md")

    pd.DataFrame(all_rows).to_csv(rows_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    pd.DataFrame(dup_rows).to_csv(dup_path, index=False)
    pd.DataFrame(sig_rows).to_csv(sig_path, index=False)
    payload = {
        "protocol": {
            "candidate": METHOD_B5,
            "threshold": args.threshold,
            "gain": args.gain,
            "shifted_cap": 1.0,
            "gate": "SciFact trainfit binary BGE-base rescue classifier",
            "c_value": 0.1,
        },
        "train_diagnostics": train_diagnostics,
        "source_feature_stats": stats,
        "dataset_diagnostics": diagnostics,
        "summary": summary,
        "duplicate_filtered_scifact": dup_rows,
        "significance": sig_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, all_rows, summary, dup_rows, sig_rows, diagnostics)
    print(json.dumps(payload["protocol"], indent=2))
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {dup_path}")
    print(f"Wrote {sig_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
