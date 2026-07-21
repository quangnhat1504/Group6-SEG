"""Phase 8C rank-window smoothing ablation for Frozen B5 SGAF.

This is a cheap post-retrieval experiment. For batches where Frozen B5 already
selects generalist-fallback mode, it reorders only a small top-rank window using
a weak BGE-small specialist prior. Source-like batches are left unchanged.

The script is exploratory: alpha/window are swept on existing runs and should
not be promoted as a final frozen recipe without a new validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np

from run_sgaf_post_retrieval_collapse_ablation import load_qrels_flexible, write_csv
from run_significance_conformal import paired_bootstrap
from seg_retrieval.io import load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.types import Run


DATASETS = ("scifact", "nfcorpus", "fiqa", "scidocs")
TRANSFER_DATASETS = ("nfcorpus", "fiqa", "scidocs")
METHOD_BGE_SMALL = "BGE-small specialist"
METHOD_BGE_BASE = "BGE-base generalist"
METHOD_CURRENT = "Current adaptive SGAF"
METHOD_B5 = "Frozen B5 mode-switch SGAF"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def by_dataset_method(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {(row["dataset"], row["method"]): row for row in rows}
    for dataset in DATASETS:
        for method in (METHOD_BGE_SMALL, METHOD_BGE_BASE, METHOD_CURRENT, METHOD_B5):
            if (dataset, method) not in out:
                raise ValueError(f"Missing row for {dataset} / {method}")
    return out


def rank_lookup(hits: list[tuple[str, float]], limit: int) -> dict[str, int]:
    return {doc_id: rank for rank, (doc_id, _) in enumerate(hits[:limit], start=1)}


def smooth_query(
    base_hits: list[tuple[str, float]],
    specialist_hits: list[tuple[str, float]],
    *,
    alpha: float,
    window: int,
    rrf_k: int,
    top_k: int,
) -> list[tuple[str, float]]:
    base_rank = rank_lookup(base_hits, window)
    specialist_rank = rank_lookup(specialist_hits, window)
    candidates = set(base_rank) | set(specialist_rank)

    scores: dict[str, float] = {}
    for doc_id in candidates:
        score = 0.0
        if doc_id in base_rank:
            score += (1.0 - alpha) / (rrf_k + base_rank[doc_id])
        if doc_id in specialist_rank:
            score += alpha / (rrf_k + specialist_rank[doc_id])
        scores[doc_id] = score

    smoothed_head = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    seen = {doc_id for doc_id, _ in smoothed_head}
    tail = [(doc_id, score) for doc_id, score in base_hits if doc_id not in seen]
    return (smoothed_head + tail)[:top_k]


def smooth_run(
    base_run: Run,
    specialist_run: Run,
    *,
    alpha: float,
    window: int,
    rrf_k: int,
    top_k: int,
) -> Run:
    return {
        query_id: smooth_query(
            base_run.get(query_id, []),
            specialist_run.get(query_id, []),
            alpha=alpha,
            window=window,
            rrf_k=rrf_k,
            top_k=top_k,
        )
        for query_id in base_run
    }


def variant_name(window: int, alpha: float) -> str:
    return f"p3_window_{window}_alpha_{alpha:.3f}".replace(".", "_")


def run_ablation(
    final_rows_path: Path,
    data_dir: Path,
    output_dir: Path,
    windows: list[int],
    alphas: list[float],
    rrf_k: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = read_csv(final_rows_path)
    rows_by_key = by_dataset_method(final_rows)
    qrels = {dataset: load_qrels_flexible(dataset, data_dir) for dataset in DATASETS}

    loaded_runs: dict[tuple[str, str], Run] = {}
    for dataset in DATASETS:
        for method in (METHOD_BGE_SMALL, METHOD_B5):
            loaded_runs[(dataset, method)] = load_run(rows_by_key[(dataset, method)]["run_path"])

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    bge_base_transfer = sum(f(rows_by_key[(dataset, METHOD_BGE_BASE)], "ndcg@10") for dataset in TRANSFER_DATASETS) / len(
        TRANSFER_DATASETS
    )
    b5_transfer = sum(f(rows_by_key[(dataset, METHOD_B5)], "ndcg@10") for dataset in TRANSFER_DATASETS) / len(
        TRANSFER_DATASETS
    )

    for window in windows:
        for alpha in alphas:
            name = variant_name(window, alpha)
            variant_metrics: dict[str, dict[str, float]] = {}
            for dataset in DATASETS:
                b5_row = rows_by_key[(dataset, METHOD_B5)]
                params = json.loads(b5_row["params"])
                mode = params.get("mode", "")

                if mode == "generalist_fallback":
                    run = smooth_run(
                        loaded_runs[(dataset, METHOD_B5)],
                        loaded_runs[(dataset, METHOD_BGE_SMALL)],
                        alpha=alpha,
                        window=window,
                        rrf_k=rrf_k,
                        top_k=top_k,
                    )
                else:
                    run = loaded_runs[(dataset, METHOD_B5)]

                run_path = output_dir / f"{dataset}_{name}.csv"
                save_run(run_path, run)
                metrics = evaluate_run(run, qrels[dataset], include_map=True)
                variant_metrics[dataset] = metrics

                detail_rows.append(
                    {
                        "variant": name,
                        "dataset": dataset,
                        "window": window,
                        "alpha": alpha,
                        "mode": mode,
                        "baseline_b5_ndcg@10": f(b5_row, "ndcg@10"),
                        "smoothed_ndcg@10": metrics["ndcg@10"],
                        "delta_vs_b5": metrics["ndcg@10"] - f(b5_row, "ndcg@10"),
                        "delta_vs_bge_base": metrics["ndcg@10"] - f(rows_by_key[(dataset, METHOD_BGE_BASE)], "ndcg@10"),
                        "delta_vs_current_adaptive": metrics["ndcg@10"] - f(
                            rows_by_key[(dataset, METHOD_CURRENT)], "ndcg@10"
                        ),
                        "recall@100": metrics["recall@100"],
                        "run_path": str(run_path),
                    }
                )

            transfer_values = [variant_metrics[dataset]["ndcg@10"] for dataset in TRANSFER_DATASETS]
            transfer_avg = sum(transfer_values) / len(transfer_values)
            deltas_vs_b5 = [
                variant_metrics[dataset]["ndcg@10"] - f(rows_by_key[(dataset, METHOD_B5)], "ndcg@10")
                for dataset in TRANSFER_DATASETS
            ]
            min_transfer_delta = min(deltas_vs_b5)
            max_transfer_delta = max(deltas_vs_b5)
            scifact_delta = variant_metrics["scifact"]["ndcg@10"] - f(rows_by_key[("scifact", METHOD_B5)], "ndcg@10")

            if transfer_avg - b5_transfer >= 0.001 and min_transfer_delta >= -0.0005 and scifact_delta >= -0.0005:
                decision = "candidate"
            elif transfer_avg > b5_transfer:
                decision = "diagnostic_positive"
            else:
                decision = "reject"

            summary_rows.append(
                {
                    "variant": name,
                    "window": window,
                    "alpha": alpha,
                    "avg_ndcg@10": sum(metric["ndcg@10"] for metric in variant_metrics.values()) / len(DATASETS),
                    "transfer_avg_ndcg@10": transfer_avg,
                    "transfer_delta_vs_b5": transfer_avg - b5_transfer,
                    "transfer_delta_vs_bge_base": transfer_avg - bge_base_transfer,
                    "min_transfer_delta_vs_b5": min_transfer_delta,
                    "max_transfer_delta_vs_b5": max_transfer_delta,
                    "scifact_delta_vs_b5": scifact_delta,
                    "decision": decision,
                }
            )

    summary_rows.sort(
        key=lambda row: (
            row["decision"] != "candidate",
            -float(row["transfer_delta_vs_b5"]),
            float(row["min_transfer_delta_vs_b5"]),
        )
    )
    return detail_rows, summary_rows


def best_variant_significance(
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    final_rows_path: Path,
    data_dir: Path,
    *,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    final_rows = read_csv(final_rows_path)
    rows_by_key = by_dataset_method(final_rows)
    best_variant = summary_rows[0]["variant"]
    best_rows = [row for row in detail_rows if row["variant"] == best_variant]
    rng = np.random.default_rng(seed)
    sig_rows: list[dict[str, Any]] = []

    for row in best_rows:
        dataset = row["dataset"]
        qrels = load_qrels_flexible(dataset, data_dir)
        system_run = load_run(row["run_path"])
        system_scores = per_query_ndcg(system_run, qrels, 10)
        query_ids = sorted(qrels)
        system_vals = np.asarray([system_scores.get(query_id, 0.0) for query_id in query_ids], dtype=np.float64)

        for baseline in (METHOD_B5, METHOD_BGE_BASE):
            baseline_run = load_run(rows_by_key[(dataset, baseline)]["run_path"])
            baseline_scores = per_query_ndcg(baseline_run, qrels, 10)
            baseline_vals = np.asarray(
                [baseline_scores.get(query_id, 0.0) for query_id in query_ids],
                dtype=np.float64,
            )
            mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(
                system_vals,
                baseline_vals,
                rng,
                n_boot=n_boot,
            )
            sig_rows.append(
                {
                    "variant": best_variant,
                    "dataset": dataset,
                    "baseline": baseline,
                    "n_queries": len(query_ids),
                    "mean_delta_ndcg@10": mean_diff,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "p_value": p_value,
                    "significant": ci_lo > 0.0 or ci_hi < 0.0,
                }
            )
    return sig_rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_signed(value: Any) -> str:
    return f"{float(value):+.4f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit is not None else rows
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key.startswith("delta") or "_delta_" in key:
                cells.append(fmt_signed(value))
            else:
                cells.append(fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    path: Path,
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    sig_rows: list[dict[str, Any]],
) -> None:
    best = summary_rows[0]
    best_detail = [row for row in detail_rows if row["variant"] == best["variant"]]
    lines = [
        "# SGAF Phase 8C Rank-Window Smoothing Ablation",
        "",
        "This cheap post-retrieval ablation blends a small BGE-small specialist prior into Frozen B5 only when the batch is already in `generalist_fallback` mode. Source-like SciFact is left unchanged.",
        "",
        "Important caveat: this is an exploratory sweep over `window` and `alpha`; do not treat the best row as a frozen final recipe without a new validation split.",
        "",
        "## Summary",
        "",
    ]
    lines += markdown_table(
        summary_rows,
        [
            ("variant", "Variant"),
            ("window", "Window"),
            ("alpha", "Alpha"),
            ("transfer_avg_ndcg@10", "Transfer Avg"),
            ("transfer_delta_vs_b5", "Delta vs B5"),
            ("transfer_delta_vs_bge_base", "Delta vs BGE-base"),
            ("min_transfer_delta_vs_b5", "Min transfer delta"),
            ("scifact_delta_vs_b5", "SciFact delta"),
            ("decision", "Decision"),
        ],
    )
    lines += [
        "",
        "## Best Variant Detail",
        "",
    ]
    lines += markdown_table(
        best_detail,
        [
            ("dataset", "Dataset"),
            ("mode", "B5 mode"),
            ("baseline_b5_ndcg@10", "B5 nDCG@10"),
            ("smoothed_ndcg@10", "Smoothed nDCG@10"),
            ("delta_vs_b5", "Delta vs B5"),
            ("delta_vs_bge_base", "Delta vs BGE-base"),
            ("delta_vs_current_adaptive", "Delta vs current"),
        ],
    )
    lines += [
        "",
        "## Best Variant Paired Bootstrap",
        "",
    ]
    lines += markdown_table(
        sig_rows,
        [
            ("dataset", "Dataset"),
            ("baseline", "Baseline"),
            ("n_queries", "Queries"),
            ("mean_delta_ndcg@10", "Mean delta"),
            ("ci_lo", "CI low"),
            ("ci_hi", "CI high"),
            ("p_value", "p-value"),
            ("significant", "Significant"),
        ],
    )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A candidate row must improve transfer average by at least `0.001`, avoid any transfer-dataset loss below `-0.0005`, and keep SciFact unchanged.",
        "- If all rows are rejected, adding the specialist prior after Frozen B5 mostly reintroduces the transfer weakness that B5 was designed to avoid.",
        "- If a row is only `diagnostic_positive`, treat it as potential failure-analysis material, not a final method.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-rows",
        type=Path,
        default=Path("runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_rows.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fusion/phase8_rank_window_smoothing"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/tables/table_sgaf_phase8_rank_window_smoothing.md"),
    )
    parser.add_argument("--windows", default="10,20")
    parser.add_argument("--alphas", default="0.025,0.05,0.10,0.15,0.20")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    windows = parse_int_list(args.windows)
    alphas = parse_float_list(args.alphas)
    detail_rows, summary_rows = run_ablation(
        args.final_rows,
        args.data_dir,
        args.output_dir,
        windows,
        alphas,
        args.rrf_k,
        args.top_k,
    )
    sig_rows = best_variant_significance(
        detail_rows,
        summary_rows,
        args.final_rows,
        args.data_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    write_csv(
        args.output_dir / "phase8_rank_window_smoothing_rows.csv",
        detail_rows,
        [
            "variant",
            "dataset",
            "window",
            "alpha",
            "mode",
            "baseline_b5_ndcg@10",
            "smoothed_ndcg@10",
            "delta_vs_b5",
            "delta_vs_bge_base",
            "delta_vs_current_adaptive",
            "recall@100",
            "run_path",
        ],
    )
    write_csv(
        args.output_dir / "phase8_rank_window_smoothing_summary.csv",
        summary_rows,
        [
            "variant",
            "window",
            "alpha",
            "avg_ndcg@10",
            "transfer_avg_ndcg@10",
            "transfer_delta_vs_b5",
            "transfer_delta_vs_bge_base",
            "min_transfer_delta_vs_b5",
            "max_transfer_delta_vs_b5",
            "scifact_delta_vs_b5",
            "decision",
        ],
    )
    write_csv(
        args.output_dir / "phase8_rank_window_smoothing_significance.csv",
        sig_rows,
        [
            "variant",
            "dataset",
            "baseline",
            "n_queries",
            "mean_delta_ndcg@10",
            "ci_lo",
            "ci_hi",
            "p_value",
            "significant",
        ],
    )
    write_report(args.report, detail_rows, summary_rows, sig_rows)

    manifest = {
        "source_artifact": str(args.final_rows),
        "outputs": {
            "rows": str(args.output_dir / "phase8_rank_window_smoothing_rows.csv"),
            "summary": str(args.output_dir / "phase8_rank_window_smoothing_summary.csv"),
            "significance": str(args.output_dir / "phase8_rank_window_smoothing_significance.csv"),
            "report": str(args.report),
        },
        "grid": {"windows": windows, "alphas": alphas, "rrf_k": args.rrf_k, "top_k": args.top_k},
        "n_boot": args.n_boot,
        "caveat": "exploratory sweep; not a frozen final recipe",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase8_rank_window_smoothing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    best = summary_rows[0]
    print(f"Wrote smoothing ablation artifacts to {args.output_dir}")
    print(f"Wrote report to {args.report}")
    print(
        "Best variant: "
        f"{best['variant']} transfer_delta_vs_b5={best['transfer_delta_vs_b5']:+.6f} "
        f"decision={best['decision']}"
    )


if __name__ == "__main__":
    main()
