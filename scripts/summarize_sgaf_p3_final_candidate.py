"""Summarize the frozen P3 rank-window smoothing candidate.

P3 is frozen from the Phase 8C/LOTO evidence:
  - window = 20
  - alpha = 0.10
  - applied only to Frozen B5 generalist-fallback batches

The script reads existing artifacts only. It does not rerun retrieval and does
not reselect the variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ("scifact", "nfcorpus", "fiqa", "scidocs")
TRANSFER_DATASETS = ("nfcorpus", "fiqa", "scidocs")

METHOD_BGE_SMALL = "BGE-small specialist"
METHOD_BGE_BASE = "BGE-base generalist"
METHOD_CURRENT = "Current adaptive SGAF"
METHOD_B5 = "Frozen B5 mode-switch SGAF"
METHOD_P3 = "Frozen P3 rank-window smoothing SGAF"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def final_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {(row["dataset"], row["method"]): row for row in rows}
    for dataset in DATASETS:
        for method in (METHOD_BGE_SMALL, METHOD_BGE_BASE, METHOD_CURRENT, METHOD_B5):
            if (dataset, method) not in out:
                raise ValueError(f"Missing final row for {dataset} / {method}")
    return out


def selected_p3_by_dataset(rows: list[dict[str, str]], variant: str) -> dict[str, dict[str, str]]:
    out = {row["dataset"]: row for row in rows if row["variant"] == variant}
    missing = sorted(set(DATASETS) - set(out))
    if missing:
        raise ValueError(f"Missing selected P3 rows for datasets: {missing}")
    return out


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method]
    if len(method_rows) != len(DATASETS):
        raise ValueError(f"Expected {len(DATASETS)} rows for {method}, found {len(method_rows)}")
    by_dataset = {row["dataset"]: row for row in method_rows}
    return {
        "method": method,
        "avg_ndcg@10": mean([float(row["ndcg@10"]) for row in method_rows]),
        "transfer_avg_ndcg@10": mean([float(by_dataset[dataset]["ndcg@10"]) for dataset in TRANSFER_DATASETS]),
        "scifact_ndcg@10": float(by_dataset["scifact"]["ndcg@10"]),
    }


def build_rows(
    final_rows: list[dict[str, str]],
    p3_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    final = final_by_key(final_rows)
    rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for method in (METHOD_BGE_SMALL, METHOD_BGE_BASE, METHOD_CURRENT, METHOD_B5):
            base_row = final[(dataset, method)]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mode": json.loads(base_row["params"]).get("mode", "component") if base_row["params"] else "component",
                    "ndcg@10": f(base_row, "ndcg@10"),
                    "recall@100": f(base_row, "recall@100"),
                    "run_path": base_row["run_path"],
                }
            )

        p3_row = p3_rows[dataset]
        rows.append(
            {
                "dataset": dataset,
                "method": METHOD_P3,
                "mode": p3_row["mode"],
                "ndcg@10": f(p3_row, "smoothed_ndcg@10"),
                "recall@100": f(p3_row, "recall@100"),
                "run_path": p3_row["run_path"],
            }
        )

    by_dataset_method = {(row["dataset"], row["method"]): row for row in rows}
    for row in rows:
        dataset = row["dataset"]
        row["delta_vs_bge_small"] = row["ndcg@10"] - by_dataset_method[(dataset, METHOD_BGE_SMALL)]["ndcg@10"]
        row["delta_vs_bge_base"] = row["ndcg@10"] - by_dataset_method[(dataset, METHOD_BGE_BASE)]["ndcg@10"]
        row["delta_vs_current_adaptive"] = row["ndcg@10"] - by_dataset_method[(dataset, METHOD_CURRENT)]["ndcg@10"]
        row["delta_vs_b5"] = row["ndcg@10"] - by_dataset_method[(dataset, METHOD_B5)]["ndcg@10"]
    return rows


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [summarize_method(rows, method) for method in (METHOD_BGE_SMALL, METHOD_BGE_BASE, METHOD_CURRENT, METHOD_B5, METHOD_P3)]
    by_method = {row["method"]: row for row in summaries}
    for row in summaries:
        row["avg_delta_vs_bge_small"] = row["avg_ndcg@10"] - by_method[METHOD_BGE_SMALL]["avg_ndcg@10"]
        row["transfer_delta_vs_bge_base"] = (
            row["transfer_avg_ndcg@10"] - by_method[METHOD_BGE_BASE]["transfer_avg_ndcg@10"]
        )
        row["transfer_delta_vs_b5"] = row["transfer_avg_ndcg@10"] - by_method[METHOD_B5]["transfer_avg_ndcg@10"]
        row["scifact_delta_vs_bge_small"] = row["scifact_ndcg@10"] - by_method[METHOD_BGE_SMALL]["scifact_ndcg@10"]
    return summaries


def build_duplicate_rows(duplicate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_method in (METHOD_BGE_SMALL, METHOD_BGE_BASE, METHOD_CURRENT, METHOD_B5):
        source = next(row for row in duplicate_rows if row["method"] == source_method)
        rows.append(
            {
                "method": source_method,
                "filtered_ndcg@10": f(source, "filtered_ndcg@10"),
                "filtered_delta_vs_bge_small": f(source, "filtered_delta_vs_bge_small"),
                "note": "measured",
            }
        )
    b5 = next(row for row in duplicate_rows if row["method"] == METHOD_B5)
    rows.append(
        {
            "method": METHOD_P3,
            "filtered_ndcg@10": f(b5, "filtered_ndcg@10"),
            "filtered_delta_vs_bge_small": f(b5, "filtered_delta_vs_bge_small"),
            "note": "same as Frozen B5; SciFact is specialist_safe and not smoothed",
        }
    )
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_signed(value: Any) -> str:
    return f"{float(value):+.4f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if key.startswith("delta") or "_delta_" in key or key.endswith("_delta_vs_bge_base") or key.endswith("_delta_vs_b5"):
                cells.append(fmt_signed(value))
            else:
                cells.append(fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    loto_rows: list[dict[str, str]],
    loto_sig_rows: list[dict[str, str]],
    *,
    variant: str,
) -> None:
    p3_summary = next(row for row in summary_rows if row["method"] == METHOD_P3)
    p3_dataset_rows = [row for row in dataset_rows if row["method"] == METHOD_P3]
    lines = [
        "# Frozen P3 Rank-Window Smoothing SGAF Candidate",
        "",
        f"Frozen recipe: use P3 variant `{variant}` (`window=20`, `alpha=0.10`, `rrf_k=60`) after Frozen B5, only on `generalist_fallback` batches.",
        "",
        "This report is a candidate synthesis from existing Phase 8 artifacts. It does not rerun retrieval and does not reselect hyperparameters.",
        "",
        "## Summary",
        "",
    ]
    lines += markdown_table(
        summary_rows,
        [
            ("method", "Method"),
            ("avg_ndcg@10", "Avg nDCG@10"),
            ("transfer_avg_ndcg@10", "Transfer Avg"),
            ("avg_delta_vs_bge_small", "Avg delta vs BGE-small"),
            ("transfer_delta_vs_bge_base", "Transfer delta vs BGE-base"),
            ("transfer_delta_vs_b5", "Transfer delta vs B5"),
            ("scifact_delta_vs_bge_small", "SciFact delta"),
        ],
    )
    lines += [
        "",
        "## Dataset Detail For P3",
        "",
    ]
    lines += markdown_table(
        p3_dataset_rows,
        [
            ("dataset", "Dataset"),
            ("mode", "B5 mode"),
            ("ndcg@10", "P3 nDCG@10"),
            ("delta_vs_bge_small", "Delta vs BGE-small"),
            ("delta_vs_bge_base", "Delta vs BGE-base"),
            ("delta_vs_current_adaptive", "Delta vs current"),
            ("delta_vs_b5", "Delta vs B5"),
            ("recall@100", "Recall@100"),
        ],
    )
    lines += [
        "",
        "## Duplicate-Filtered SciFact",
        "",
    ]
    lines += markdown_table(
        duplicate_rows,
        [
            ("method", "Method"),
            ("filtered_ndcg@10", "Filtered nDCG@10"),
            ("filtered_delta_vs_bge_small", "Filtered delta vs BGE-small"),
            ("note", "Note"),
        ],
    )
    lines += [
        "",
        "## LOTO Evidence",
        "",
    ]
    lines += markdown_table(
        [
            {
                "heldout_dataset": row["heldout_dataset"],
                "selected_variant": row["selected_variant"],
                "heldout_delta_vs_b5": float(row["heldout_delta_vs_b5"]),
                "heldout_delta_vs_bge_base": float(row["heldout_delta_vs_bge_base"]),
            }
            for row in loto_rows
        ],
        [
            ("heldout_dataset", "Held-out"),
            ("selected_variant", "Selected variant"),
            ("heldout_delta_vs_b5", "Delta vs B5"),
            ("heldout_delta_vs_bge_base", "Delta vs BGE-base"),
        ],
    )
    lines += [
        "",
        "## Paired Bootstrap For LOTO Held-Out",
        "",
    ]
    lines += markdown_table(
        [
            {
                "heldout_dataset": row["heldout_dataset"],
                "baseline": row["baseline"],
                "mean_delta_ndcg@10": float(row["mean_delta_ndcg@10"]),
                "ci_lo": float(row["ci_lo"]),
                "ci_hi": float(row["ci_hi"]),
                "p_value": float(row["p_value"]),
                "significant": row["significant"],
            }
            for row in loto_sig_rows
        ],
        [
            ("heldout_dataset", "Held-out"),
            ("baseline", "Baseline"),
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
        f"- Frozen P3 raises transfer average to `{p3_summary['transfer_avg_ndcg@10']:.4f}`, above Frozen B5 and BGE-base on the current evaluated transfer datasets.",
        "- LOTO selection chooses the same variant in all three transfer folds, so the signal is not driven by a single held-out dataset.",
        "- Keep the claim caveated: the P3 grid was designed after seeing the project datasets, so external frozen validation is still required for a paper-grade claim.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="p3_window_20_alpha_0_100")
    parser.add_argument(
        "--final-rows",
        type=Path,
        default=Path("runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_rows.csv"),
    )
    parser.add_argument(
        "--duplicate-filtered",
        type=Path,
        default=Path("runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_duplicate_filtered_scifact.csv"),
    )
    parser.add_argument(
        "--p3-rows",
        type=Path,
        default=Path("runs/fusion/phase8_rank_window_smoothing/phase8_rank_window_smoothing_rows.csv"),
    )
    parser.add_argument(
        "--loto-rows",
        type=Path,
        default=Path("runs/fusion/phase8_rank_window_smoothing/phase8_p3_loto_rows.csv"),
    )
    parser.add_argument(
        "--loto-significance",
        type=Path,
        default=Path("runs/fusion/phase8_rank_window_smoothing/phase8_p3_loto_significance.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fusion/final_sgaf_p3_smoothing"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/tables/table_final_sgaf_p3_smoothing.md"),
    )
    args = parser.parse_args()

    final_rows = read_csv(args.final_rows)
    selected_p3_rows = selected_p3_by_dataset(read_csv(args.p3_rows), args.variant)
    dataset_rows = build_rows(final_rows, selected_p3_rows)
    summary_rows = build_summary(dataset_rows)
    duplicate_rows = build_duplicate_rows(read_csv(args.duplicate_filtered))
    loto_rows = read_csv(args.loto_rows)
    loto_sig_rows = read_csv(args.loto_significance)

    write_csv(
        args.output_dir / "final_sgaf_p3_smoothing_rows.csv",
        dataset_rows,
        [
            "dataset",
            "method",
            "mode",
            "ndcg@10",
            "recall@100",
            "run_path",
            "delta_vs_bge_small",
            "delta_vs_bge_base",
            "delta_vs_current_adaptive",
            "delta_vs_b5",
        ],
    )
    write_csv(
        args.output_dir / "final_sgaf_p3_smoothing_summary.csv",
        summary_rows,
        [
            "method",
            "avg_ndcg@10",
            "transfer_avg_ndcg@10",
            "scifact_ndcg@10",
            "avg_delta_vs_bge_small",
            "transfer_delta_vs_bge_base",
            "transfer_delta_vs_b5",
            "scifact_delta_vs_bge_small",
        ],
    )
    write_csv(
        args.output_dir / "final_sgaf_p3_smoothing_duplicate_filtered_scifact.csv",
        duplicate_rows,
        ["method", "filtered_ndcg@10", "filtered_delta_vs_bge_small", "note"],
    )
    write_report(
        args.report,
        summary_rows,
        dataset_rows,
        duplicate_rows,
        loto_rows,
        loto_sig_rows,
        variant=args.variant,
    )

    manifest = {
        "variant": args.variant,
        "recipe": {
            "window": 20,
            "alpha": 0.10,
            "rrf_k": 60,
            "apply_only_when_b5_mode": "generalist_fallback",
        },
        "source_artifacts": {
            "final_rows": str(args.final_rows),
            "p3_rows": str(args.p3_rows),
            "loto_rows": str(args.loto_rows),
            "loto_significance": str(args.loto_significance),
            "duplicate_filtered": str(args.duplicate_filtered),
        },
        "outputs": {
            "rows": str(args.output_dir / "final_sgaf_p3_smoothing_rows.csv"),
            "summary": str(args.output_dir / "final_sgaf_p3_smoothing_summary.csv"),
            "duplicate_filtered": str(
                args.output_dir / "final_sgaf_p3_smoothing_duplicate_filtered_scifact.csv"
            ),
            "report": str(args.report),
        },
        "caveat": "candidate is frozen from Phase 8 LOTO evidence but still needs external/new-batch validation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_sgaf_p3_smoothing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    p3_summary = next(row for row in summary_rows if row["method"] == METHOD_P3)
    print(f"Wrote frozen P3 candidate artifacts to {args.output_dir}")
    print(f"Wrote report to {args.report}")
    print(
        f"Frozen P3 transfer_avg={p3_summary['transfer_avg_ndcg@10']:.6f}; "
        f"delta_vs_B5={p3_summary['transfer_delta_vs_b5']:+.6f}; "
        f"delta_vs_BGE-base={p3_summary['transfer_delta_vs_bge_base']:+.6f}"
    )


if __name__ == "__main__":
    main()
