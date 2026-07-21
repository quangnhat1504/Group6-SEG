"""Paired bootstrap significance checks for retrieval runs.

This module is intentionally small and script-friendly. Tests import
``paired_bootstrap`` directly, while the CLI compares two run files using
per-query nDCG@10 and writes JSON/CSV/Markdown artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.metrics import per_query_ndcg


def paired_bootstrap(
    system_vals: np.ndarray,
    baseline_vals: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Return observed mean diff, percentile CI, and two-sided bootstrap p-value."""
    system_vals = np.asarray(system_vals, dtype=np.float64)
    baseline_vals = np.asarray(baseline_vals, dtype=np.float64)
    if system_vals.shape != baseline_vals.shape:
        raise ValueError("system_vals and baseline_vals must have the same shape")
    if system_vals.ndim != 1 or system_vals.size < 2:
        raise ValueError("paired_bootstrap expects one-dimensional arrays with n >= 2")
    if not np.all(np.isfinite(system_vals)) or not np.all(np.isfinite(baseline_vals)):
        raise ValueError("paired_bootstrap inputs must be finite")

    diffs = system_vals - baseline_vals
    mean_diff = float(np.mean(diffs))
    sample_indices = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_means = diffs[sample_indices].mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    ci_lo = float(min(ci_lo, mean_diff))
    ci_hi = float(max(ci_hi, mean_diff))

    if mean_diff >= 0:
        tail = np.mean(boot_means <= 0.0)
    else:
        tail = np.mean(boot_means >= 0.0)
    p_value = float(min(1.0, max(0.0, 2.0 * tail)))
    return mean_diff, ci_lo, ci_hi, p_value


def write_markdown(path: Path, row: dict) -> None:
    lines = [
        "# Paired Bootstrap Significance",
        "",
        "| Comparison | Queries | Mean delta nDCG@10 | 95% CI | p-value | Significant |",
        "|---|---:|---:|---:|---:|---|",
        (
            f"| {row['system']} vs {row['baseline']} | {row['n_queries']} | "
            f"{row['mean_diff']:+.6f} | [{row['ci_lo']:+.6f}, {row['ci_hi']:+.6f}] | "
            f"{row['p_value']:.4f} | {row['significant']} |"
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--system-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--system-name", default="system")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.qrels:
        qrels_path = Path(args.qrels)
    elif args.split:
        qrels_path = config.dataset.data_dir / f"{args.split}_qrels.csv"
    else:
        raise ValueError("Provide either --qrels or --split")

    qrels = load_qrels(qrels_path)
    system_run = load_run(args.system_run)
    baseline_run = load_run(args.baseline_run)
    system_scores = per_query_ndcg(system_run, qrels, 10)
    baseline_scores = per_query_ndcg(baseline_run, qrels, 10)
    query_ids = sorted(qrels)
    system_vals = np.asarray([system_scores.get(query_id, 0.0) for query_id in query_ids], dtype=np.float64)
    baseline_vals = np.asarray([baseline_scores.get(query_id, 0.0) for query_id in query_ids], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(
        system_vals,
        baseline_vals,
        rng,
        n_boot=args.n_boot,
    )
    row = {
        "system": args.system_name,
        "baseline": args.baseline_name,
        "qrels": str(qrels_path),
        "n_queries": len(query_ids),
        "system_mean": float(np.mean(system_vals)),
        "baseline_mean": float(np.mean(baseline_vals)),
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "significant": ci_lo > 0.0 or ci_hi < 0.0,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    write_markdown(md_path, row)
    print(json.dumps(row, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
