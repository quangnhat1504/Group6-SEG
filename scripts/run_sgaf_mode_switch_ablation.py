"""Explore BGE-base mode-switch coverage for SGAF.

This script starts Phase 7 with a narrow diagnostic:

- B4a: raise the max coverage cap while keeping the current uncertainty formula.
- B4b: increase the uncertainty gain and allow larger caps.
- B5: enable high gain only for batches whose cheap source-shift score is high.

The goal is to separate two possible causes of the BGE-base transfer gap:

1. the hard coverage cap is too low;
2. the uncertainty formula is too conservative before it ever reaches the cap.

Target qrels are used only for evaluation and oracle-style analysis. The gain
sweep is exploratory and should not be reported as a frozen final claim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from evaluate_frozen_sgaf_transfer import DATASETS, load_query_texts_and_qrels, load_runs, load_train_state, select_routes
from run_adaptive_coverage_sgaf import source_feature_stats, target_feature_means, zscores
from run_significance_conformal import paired_bootstrap
from seg_retrieval.io import load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from train_query_adaptive_fusion import BASELINE, COMPONENTS, route_run


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def uncertainty_coverage(z: dict[str, float], *, gain: float, cap: float, floor: float = 0.02) -> float:
    """Current uncertainty formula with scalable non-floor terms."""
    surplus = 0.08 * max(0.0, -z["bge_small_gap"]) + 0.04 * max(0.0, -z["bge_small_std10"])
    return clamp(0.05 + gain * surplus, floor, cap)


def batch_shift_score(z: dict[str, float]) -> float:
    """Cheap batch-level shift score against SciFact trainfit feature statistics."""
    return float(
        abs(z["query_len"])
        + max(0.0, -z["bge_small_top"])
        + max(0.0, -z["bge_small_gap"])
        + max(0.0, -z["bge_small_std10"])
        + max(0.0, -z["bge_small_bge_base_overlap10"])
    )


def batch_mode_coverage(
    z: dict[str, float],
    *,
    shift_threshold: float,
    shifted_gain: float,
    shifted_cap: float = 1.0,
) -> tuple[float, str, float]:
    """Use current coverage for source-like batches and high gain for shifted batches."""
    score = batch_shift_score(z)
    if score >= shift_threshold:
        return uncertainty_coverage(z, gain=shifted_gain, cap=shifted_cap), "generalist_fallback", score
    return uncertainty_coverage(z, gain=1.0, cap=0.4), "specialist_safe", score


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")


def evaluate_variant(
    *,
    dataset: str,
    method: str,
    stage: str,
    coverage: float,
    model,
    class_index: int,
    runs,
    query_texts,
    qrels,
    output_dir: Path,
    params: dict,
) -> tuple[dict, dict]:
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
    path = output_dir / f"{dataset}_{safe_slug(method)}.csv"
    save_run(path, run)
    metrics = evaluate_run(run, qrels, include_map=True)
    return (
        {
            "dataset": dataset,
            "stage": stage,
            "method": method,
            "coverage": coverage,
            "switch_rate": sum(route != BASELINE for route in routes.values()) / len(routes),
            **metrics,
            "params": json.dumps(params, sort_keys=True),
            "run_path": str(path),
        },
        run,
    )


def component_rows(dataset: str, runs, qrels) -> list[dict]:
    rows = []
    for component in COMPONENTS:
        metrics = evaluate_run(runs[component], qrels, include_map=True)
        rows.append(
            {
                "dataset": dataset,
                "stage": f"B:{component}",
                "method": f"Component {component}",
                "coverage": None,
                "switch_rate": None,
                **metrics,
                "params": json.dumps({}),
                "run_path": "",
            }
        )
    return rows


def add_comparison_columns(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    by_dataset = {}
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        by_dataset[dataset] = {
            "bge_small": float(subset[subset["method"] == f"Component {BASELINE}"]["ndcg@10"].iloc[0]),
            "bge_base": float(subset[subset["method"] == "Component bge_base"]["ndcg@10"].iloc[0]),
            "current": float(subset[subset["method"] == "Current uncertainty coverage"]["ndcg@10"].iloc[0]),
        }
    previous_by_dataset: dict[str, float] = {}
    for row in rows:
        refs = by_dataset[row["dataset"]]
        row["delta_vs_bge_small"] = row["ndcg@10"] - refs["bge_small"]
        row["delta_vs_bge_base"] = row["ndcg@10"] - refs["bge_base"]
        row["delta_vs_current_adaptive"] = row["ndcg@10"] - refs["current"]
        row["delta_vs_previous"] = row["ndcg@10"] - previous_by_dataset.get(row["dataset"], row["ndcg@10"])
        previous_by_dataset[row["dataset"]] = row["ndcg@10"]


def summarize_methods(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    out = []
    for method, subset in df.groupby("method", sort=False):
        transfer = subset[subset["dataset"] != "scifact"]
        out.append(
            {
                "method": method,
                "avg_ndcg@10": float(subset["ndcg@10"].mean()),
                "transfer_avg_ndcg@10": float(transfer["ndcg@10"].mean()) if not transfer.empty else float("nan"),
                "avg_delta_vs_bge_small": float(subset["delta_vs_bge_small"].mean()),
                "transfer_delta_vs_bge_base": float(transfer["delta_vs_bge_base"].mean()) if not transfer.empty else float("nan"),
                "scifact_delta_vs_bge_small": float(subset[subset["dataset"] == "scifact"]["delta_vs_bge_small"].iloc[0])
                if not subset[subset["dataset"] == "scifact"].empty
                else float("nan"),
            }
        )
    return out


def significance(rows: list[dict], output_dir: Path, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    sig = []
    candidate_methods = [
        "Current uncertainty coverage",
        "Cap-only max 1.00",
        "Gain 2.0 max 1.00",
        "Gain 4.0 max 1.00",
        "Gain 8.0 max 1.00",
        "Batch shift t2.0 gain 4.0",
        "Batch shift t2.0 gain 6.0",
        "Batch shift t2.0 gain 8.0",
    ]
    for dataset, spec in DATASETS.items():
        _, qrels = load_query_texts_and_qrels(spec)
        run_dir = Path(spec["run_dir"])
        bge_base_run = load_run(run_dir / spec["runs"]["bge_base"])
        current_path = output_dir / f"{dataset}_{safe_slug('Current uncertainty coverage')}.csv"
        current_run = load_run(current_path)
        baselines = {
            "current adaptive": current_run,
            "bge_base": bge_base_run,
        }
        for method in candidate_methods:
            path = output_dir / f"{dataset}_{safe_slug(method)}.csv"
            if not path.exists():
                continue
            system_run = load_run(path)
            for baseline_name, baseline_run in baselines.items():
                if method == "Current uncertainty coverage" and baseline_name == "current adaptive":
                    continue
                system_scores = per_query_ndcg(system_run, qrels, 10)
                baseline_scores = per_query_ndcg(baseline_run, qrels, 10)
                query_ids = sorted(qrels)
                system_vals = np.asarray([system_scores.get(qid, 0.0) for qid in query_ids])
                baseline_vals = np.asarray([baseline_scores.get(qid, 0.0) for qid in query_ids])
                mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(system_vals, baseline_vals, rng, n_boot=n_boot)
                sig.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": baseline_name,
                        "mean_delta_ndcg@10": mean_diff,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "p_value": p_value,
                        "significant": ci_lo > 0 or ci_hi < 0,
                    }
                )
    return sig


def write_markdown(path: Path, rows: list[dict], method_summary: list[dict], sig: list[dict]) -> None:
    df = pd.DataFrame(rows)
    summary = pd.DataFrame(method_summary)
    lines = [
        "# SGAF BGE-base Mode-Switch Ablation",
        "",
        "This is a Phase 7 diagnostic. It tests whether the BGE-base transfer gap is caused by the hard coverage cap or by the current uncertainty formula being too conservative.",
        "",
        "## Method Summary",
        "",
        "| Method | Avg nDCG@10 | Transfer Avg | Avg delta vs BGE-small | Transfer delta vs BGE-base | SciFact delta vs BGE-small |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['method']} | {row['avg_ndcg@10']:.4f} | {row['transfer_avg_ndcg@10']:.4f} | "
            f"{row['avg_delta_vs_bge_small']:+.4f} | {row['transfer_delta_vs_bge_base']:+.4f} | "
            f"{row['scifact_delta_vs_bge_small']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Detail",
            "",
            "| Dataset | Stage | Method | Coverage | nDCG@10 | Delta vs BGE-small | Delta vs BGE-base | Delta vs current | Recall@100 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    keep = [
        f"Component {BASELINE}",
        "Component bge_base",
        "Fixed A3 coverage 5%",
        "Current uncertainty coverage",
        "Cap-only max 1.00",
        "Gain 2.0 max 1.00",
        "Gain 4.0 max 1.00",
        "Gain 8.0 max 1.00",
        "Batch shift t2.0 gain 4.0",
        "Batch shift t2.0 gain 6.0",
        "Batch shift t2.0 gain 8.0",
    ]
    for dataset in DATASETS:
        subset = df[(df["dataset"] == dataset) & (df["method"].isin(keep))].copy()
        subset["_order"] = subset["method"].map({method: index for index, method in enumerate(keep)})
        subset = subset.sort_values("_order")
        for _, row in subset.iterrows():
            coverage = "N/A" if pd.isna(row["coverage"]) else f"{row['coverage']:.3f}"
            lines.append(
                f"| {dataset} | {row['stage']} | {row['method']} | {coverage} | {row['ndcg@10']:.4f} | "
                f"{row['delta_vs_bge_small']:+.4f} | {row['delta_vs_bge_base']:+.4f} | "
                f"{row['delta_vs_current_adaptive']:+.4f} | {row['recall@100']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Significance",
            "",
            "| Dataset | Method | Baseline | Mean delta | 95% CI | p-value | Significant |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sig:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['baseline']} | {row['mean_delta_ndcg@10']:+.6f} | "
            f"[{row['ci_lo']:+.6f}, {row['ci_hi']:+.6f}] | {row['p_value']:.4f} | "
            f"{'yes' if row['significant'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If cap-only rows equal current uncertainty coverage, the hard cap is not the immediate blocker.",
            "- If gain rows improve transfer but hurt SciFact, Phase 7 should use a mode switch rather than a global gain increase.",
            "- Batch shift rows test that mode-switch hypothesis by keeping source-like batches at current coverage and increasing coverage only for shifted batches.",
            "- Gain rows are exploratory diagnostics; they use target evaluation to understand the failure mode and should not be promoted as a frozen final method.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--output-dir", default="runs/fusion/sgaf_mode_switch_ablation")
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, class_index, train_diagnostics = load_train_state(args.config, 0.1)
    stats = source_feature_stats(args.config)

    cap_only = [0.4, 0.6, 0.8, 1.0]
    gain_sweep = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    batch_modes = [
        {"threshold": 2.0, "gain": 4.0},
        {"threshold": 2.0, "gain": 6.0},
        {"threshold": 2.0, "gain": 8.0},
        {"threshold": 3.0, "gain": 4.0},
        {"threshold": 3.0, "gain": 6.0},
        {"threshold": 4.0, "gain": 6.0},
    ]
    rows: list[dict] = []
    diagnostics: dict[str, dict] = {}

    for dataset, spec in DATASETS.items():
        query_texts, qrels = load_query_texts_and_qrels(spec)
        runs = load_runs(spec)
        query_ids = sorted(qrels)
        means = target_feature_means(runs, query_texts, query_ids)
        z = zscores(means, stats)
        diagnostics[dataset] = {"feature_means": means, "z_scores": z, "batch_shift_score": batch_shift_score(z)}
        rows.extend(component_rows(dataset, runs, qrels))

        fixed_row, _ = evaluate_variant(
            dataset=dataset,
            method="Fixed A3 coverage 5%",
            stage="B2",
            coverage=0.05,
            model=model,
            class_index=class_index,
            runs=runs,
            query_texts=query_texts,
            qrels=qrels,
            output_dir=output_dir,
            params={"coverage": 0.05},
        )
        rows.append(fixed_row)

        current_coverage = uncertainty_coverage(z, gain=1.0, cap=0.4)
        current_row, _ = evaluate_variant(
            dataset=dataset,
            method="Current uncertainty coverage",
            stage="B3",
            coverage=current_coverage,
            model=model,
            class_index=class_index,
            runs=runs,
            query_texts=query_texts,
            qrels=qrels,
            output_dir=output_dir,
            params={"gain": 1.0, "cap": 0.4, "z_scores": z},
        )
        rows.append(current_row)

        for cap in cap_only:
            method = f"Cap-only max {cap:.2f}"
            coverage = uncertainty_coverage(z, gain=1.0, cap=cap)
            row, _ = evaluate_variant(
                dataset=dataset,
                method=method,
                stage="B4a",
                coverage=coverage,
                model=model,
                class_index=class_index,
                runs=runs,
                query_texts=query_texts,
                qrels=qrels,
                output_dir=output_dir,
                params={"gain": 1.0, "cap": cap, "z_scores": z},
            )
            rows.append(row)

        for gain in gain_sweep:
            method = f"Gain {gain:.1f} max 1.00"
            coverage = uncertainty_coverage(z, gain=gain, cap=1.0)
            row, _ = evaluate_variant(
                dataset=dataset,
                method=method,
                stage="B4b",
                coverage=coverage,
                model=model,
                class_index=class_index,
                runs=runs,
                query_texts=query_texts,
                qrels=qrels,
                output_dir=output_dir,
                params={"gain": gain, "cap": 1.0, "z_scores": z},
            )
            rows.append(row)

        for spec_mode in batch_modes:
            threshold = spec_mode["threshold"]
            gain = spec_mode["gain"]
            method = f"Batch shift t{threshold:.1f} gain {gain:.1f}"
            coverage, mode, score = batch_mode_coverage(z, shift_threshold=threshold, shifted_gain=gain)
            row, _ = evaluate_variant(
                dataset=dataset,
                method=method,
                stage="B5",
                coverage=coverage,
                model=model,
                class_index=class_index,
                runs=runs,
                query_texts=query_texts,
                qrels=qrels,
                output_dir=output_dir,
                params={
                    "threshold": threshold,
                    "shifted_gain": gain,
                    "shifted_cap": 1.0,
                    "mode": mode,
                    "batch_shift_score": score,
                    "z_scores": z,
                },
            )
            rows.append(row)

    add_comparison_columns(rows)
    method_summary = summarize_methods(rows)
    sig = significance(rows, output_dir, args.n_boot, args.seed)

    rows_path = output_dir / "sgaf_mode_switch_ablation_rows.csv"
    summary_path = output_dir / "sgaf_mode_switch_ablation_summary.csv"
    sig_path = output_dir / "sgaf_mode_switch_ablation_significance.csv"
    json_path = output_dir / "sgaf_mode_switch_ablation_summary.json"
    md_path = Path("reports/tables/table_sgaf_mode_switch_ablation.md")

    pd.DataFrame(rows).to_csv(rows_path, index=False)
    pd.DataFrame(method_summary).to_csv(summary_path, index=False)
    pd.DataFrame(sig).to_csv(sig_path, index=False)
    payload = {
        "protocol": {
            "phase": "B4/B5 cap-gain and batch mode-switch diagnostic",
            "gate": "SciFact trainfit binary BGE-base rescue classifier",
            "c_value": 0.1,
            "target_qrels_usage": "evaluation only; gain sweep is exploratory",
        },
        "train_diagnostics": train_diagnostics,
        "source_feature_stats": stats,
        "dataset_diagnostics": diagnostics,
        "rows": rows,
        "method_summary": method_summary,
        "significance": sig,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, rows, method_summary, sig)

    print(json.dumps(payload["protocol"], indent=2))
    print(f"Wrote {rows_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {sig_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
