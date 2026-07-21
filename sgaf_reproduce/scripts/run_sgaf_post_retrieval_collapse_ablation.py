"""Phase 8C cheap post-retrieval duplicate/canonical collapse ablation.

The ablation is label-free at transformation time: for each query, keep the
first ranked hit for each canonical document text and remove later duplicates.
It is meant to test whether post-retrieval evidence redundancy explains any
remaining BGE-base/B5 performance gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from seg_retrieval.io import load_run, save_run
from seg_retrieval.metrics import evaluate_run
from seg_retrieval.types import Qrels, Run


DATASETS = ("scifact", "nfcorpus", "fiqa", "scidocs")
METHODS = (
    "BGE-small specialist",
    "BGE-base generalist",
    "Current adaptive SGAF",
    "Frozen B5 mode-switch SGAF",
)


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def document_id(row: dict[str, Any]) -> str:
    if "doc_id" in row:
        return str(row["doc_id"])
    if "_id" in row:
        return str(row["_id"])
    raise ValueError(f"Document row has no doc id field: {row.keys()}")


def document_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "")
    body = str(row.get("abstract") or row.get("text") or "")
    return normalize_text(f"{title} {body}")


def load_canonical_map(dataset: str, data_dir: Path) -> dict[str, str]:
    dataset_dir = data_dir / dataset
    docs_path = dataset_dir / "test_documents.jsonl"
    if not docs_path.exists():
        docs_path = dataset_dir / "corpus.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"Cannot find documents for {dataset} under {dataset_dir}")

    canonical: dict[str, str] = {}
    for row in load_jsonl(docs_path):
        doc_id = document_id(row)
        key = document_text(row) or doc_id
        canonical[doc_id] = key
    return canonical


def load_qrels_flexible(dataset: str, data_dir: Path) -> Qrels:
    candidates = [
        data_dir / dataset / "test_qrels.csv",
        data_dir / dataset / "qrels" / "test.tsv",
        data_dir / dataset / dataset / "qrels" / "test.tsv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            rows = read_csv(path)
            return {
                str(row["query_id"]): {
                    str(inner["doc_id"]): int(inner["relevance"])
                    for inner in rows
                    if str(inner["query_id"]) == str(row["query_id"])
                }
                for row in rows
            }
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            qrels: Qrels = {}
            for row in reader:
                qid = str(row.get("query-id") or row.get("query_id"))
                doc_id = str(row.get("corpus-id") or row.get("doc_id"))
                rel = int(float(row.get("score") or row.get("relevance") or 0))
                qrels.setdefault(qid, {})[doc_id] = rel
            return qrels
    raise FileNotFoundError(f"Cannot find qrels for {dataset}")


def collapse_run(run: Run, canonical_map: dict[str, str]) -> tuple[Run, dict[str, int]]:
    collapsed: Run = {}
    stats = {
        "input_hits": 0,
        "output_hits": 0,
        "removed_hits": 0,
        "doc_id_duplicate_hits": 0,
        "canonical_duplicate_hits": 0,
        "affected_queries": 0,
        "queries": len(run),
    }

    for query_id, hits in run.items():
        seen_doc_ids: set[str] = set()
        seen_canonical: set[str] = set()
        output_hits = []
        removed_for_query = 0

        for doc_id, score in hits:
            stats["input_hits"] += 1
            key = canonical_map.get(doc_id, doc_id)
            is_doc_duplicate = doc_id in seen_doc_ids
            is_canonical_duplicate = key in seen_canonical
            if is_doc_duplicate:
                stats["doc_id_duplicate_hits"] += 1
            if is_canonical_duplicate:
                stats["canonical_duplicate_hits"] += 1

            if is_doc_duplicate or is_canonical_duplicate:
                stats["removed_hits"] += 1
                removed_for_query += 1
                continue

            seen_doc_ids.add(doc_id)
            seen_canonical.add(key)
            output_hits.append((doc_id, score))

        if removed_for_query:
            stats["affected_queries"] += 1
        stats["output_hits"] += len(output_hits)
        collapsed[query_id] = output_hits

    return collapsed, stats


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def run_ablation(final_rows_path: Path, data_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = read_csv(final_rows_path)
    target_rows = [
        row
        for row in final_rows
        if row.get("dataset") in DATASETS and row.get("method") in METHODS and row.get("run_path")
    ]

    canonical_maps = {dataset: load_canonical_map(dataset, data_dir) for dataset in DATASETS}
    qrels = {dataset: load_qrels_flexible(dataset, data_dir) for dataset in DATASETS}

    rows: list[dict[str, Any]] = []
    for row in target_rows:
        dataset = row["dataset"]
        method = row["method"]
        run_path = Path(row["run_path"])
        run = load_run(run_path)
        collapsed, stats = collapse_run(run, canonical_maps[dataset])
        metrics = evaluate_run(collapsed, qrels[dataset], include_map=True)

        safe_method = method.lower().replace(" ", "_").replace("-", "_")
        collapsed_path = output_dir / f"{dataset}_{safe_method}_canonical_collapse.csv"
        save_run(collapsed_path, collapsed)

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "input_hits": stats["input_hits"],
                "output_hits": stats["output_hits"],
                "removed_hits": stats["removed_hits"],
                "affected_queries": stats["affected_queries"],
                "doc_id_duplicate_hits": stats["doc_id_duplicate_hits"],
                "canonical_duplicate_hits": stats["canonical_duplicate_hits"],
                "baseline_ndcg@10": f(row, "ndcg@10"),
                "collapsed_ndcg@10": metrics["ndcg@10"],
                "delta_ndcg@10": metrics["ndcg@10"] - f(row, "ndcg@10"),
                "baseline_recall@100": f(row, "recall@100"),
                "collapsed_recall@100": metrics["recall@100"],
                "delta_recall@100": metrics["recall@100"] - f(row, "recall@100"),
                "collapsed_run_path": str(collapsed_path),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        total_removed = sum(int(row["removed_hits"]) for row in method_rows)
        mean_delta = sum(float(row["delta_ndcg@10"]) for row in method_rows) / len(method_rows)
        max_abs_delta = max(abs(float(row["delta_ndcg@10"])) for row in method_rows)
        min_delta = min(float(row["delta_ndcg@10"]) for row in method_rows)
        if total_removed == 0:
            decision = "no-op"
        elif mean_delta >= 0.001 and min_delta >= -0.0001:
            decision = "candidate"
        else:
            decision = "diagnostic_only"
        out.append(
            {
                "method": method,
                "datasets": len(method_rows),
                "total_removed_hits": total_removed,
                "total_affected_queries": sum(int(row["affected_queries"]) for row in method_rows),
                "mean_delta_ndcg@10": mean_delta,
                "max_abs_delta_ndcg@10": max_abs_delta,
                "min_delta_ndcg@10": min_delta,
                "decision": decision,
            }
        )
    return out


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
            if key.startswith("delta") or "_delta_" in key or key.startswith("mean_delta"):
                cells.append(fmt_signed(value))
            else:
                cells.append(fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SGAF Phase 8C Duplicate/Canonical Collapse Ablation",
        "",
        "This is a cheap post-retrieval ablation. It removes repeated `doc_id` or repeated normalized document text within each query ranking, keeping the highest-ranked occurrence.",
        "",
        "## Summary",
        "",
    ]
    lines += markdown_table(
        summary_rows,
        [
            ("method", "Method"),
            ("datasets", "Datasets"),
            ("total_removed_hits", "Removed hits"),
            ("total_affected_queries", "Affected queries"),
            ("mean_delta_ndcg@10", "Mean delta nDCG@10"),
            ("max_abs_delta_ndcg@10", "Max abs delta"),
            ("min_delta_ndcg@10", "Min delta"),
            ("decision", "Decision"),
        ],
    )
    lines += [
        "",
        "## Dataset Detail",
        "",
    ]
    lines += markdown_table(
        rows,
        [
            ("dataset", "Dataset"),
            ("method", "Method"),
            ("removed_hits", "Removed hits"),
            ("affected_queries", "Affected queries"),
            ("baseline_ndcg@10", "Baseline nDCG@10"),
            ("collapsed_ndcg@10", "Collapsed nDCG@10"),
            ("delta_ndcg@10", "Delta nDCG@10"),
            ("delta_recall@100", "Delta Recall@100"),
        ],
    )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If all rows are no-op, duplicate/canonical collapse is not a BEIR benchmark improvement source.",
        "- The same logic can still be useful for the production web app if retrieval returns multiple chunks from one canonical source.",
        "- Treat `diagnostic_only` rows as evidence-cleaning or UI-diversity candidates, not as Frozen B5 performance contributions.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-rows",
        type=Path,
        default=Path("runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_rows.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fusion/phase8_post_retrieval"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/tables/table_sgaf_phase8_post_retrieval_collapse.md"),
    )
    args = parser.parse_args()

    rows = run_ablation(args.final_rows, args.data_dir, args.output_dir)
    summary_rows = summarize(rows)

    write_csv(
        args.output_dir / "phase8_post_retrieval_collapse_rows.csv",
        rows,
        [
            "dataset",
            "method",
            "input_hits",
            "output_hits",
            "removed_hits",
            "affected_queries",
            "doc_id_duplicate_hits",
            "canonical_duplicate_hits",
            "baseline_ndcg@10",
            "collapsed_ndcg@10",
            "delta_ndcg@10",
            "baseline_recall@100",
            "collapsed_recall@100",
            "delta_recall@100",
            "collapsed_run_path",
        ],
    )
    write_csv(
        args.output_dir / "phase8_post_retrieval_collapse_summary.csv",
        summary_rows,
        [
            "method",
            "datasets",
            "total_removed_hits",
            "total_affected_queries",
            "mean_delta_ndcg@10",
            "max_abs_delta_ndcg@10",
            "min_delta_ndcg@10",
            "decision",
        ],
    )
    write_report(args.report, rows, summary_rows)

    manifest = {
        "source_artifact": str(args.final_rows),
        "outputs": {
            "rows": str(args.output_dir / "phase8_post_retrieval_collapse_rows.csv"),
            "summary": str(args.output_dir / "phase8_post_retrieval_collapse_summary.csv"),
            "report": str(args.report),
        },
        "method": "per-query doc_id and normalized-title-text duplicate collapse",
    }
    (args.output_dir / "phase8_post_retrieval_collapse_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    removed = sum(int(row["removed_hits"]) for row in rows)
    max_delta = max(abs(float(row["delta_ndcg@10"])) for row in rows) if rows else 0.0
    print(f"Wrote collapse ablation artifacts to {args.output_dir}")
    print(f"Wrote report to {args.report}")
    print(f"Removed hits: {removed}; max abs delta nDCG@10: {max_delta:.6f}")


if __name__ == "__main__":
    main()
