from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_run


def normalize_label(value: str) -> str:
    text = str(value).strip().lower()
    if "bm25" in text:
        return "bm25"
    if "dense" in text:
        return "dense"
    if "hybrid" in text:
        return "hybrid"
    return text


def top_docs(hits: list[tuple[str, float]], k: int) -> set[str]:
    return {doc_id for doc_id, _ in hits[:k]}


def score_gap(hits: list[tuple[str, float]]) -> float:
    if not hits:
        return 0.0
    if len(hits) == 1:
        return float(hits[0][1])
    return float(hits[0][1] - hits[1][1])


def overlap_ratio(left: list[tuple[str, float]], right: list[tuple[str, float]], k: int) -> float:
    left_docs = top_docs(left, k)
    right_docs = top_docs(right, k)
    if not left_docs and not right_docs:
        return 0.0
    return len(left_docs & right_docs) / max(len(left_docs | right_docs), 1)


def load_optional_predictions(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    df = pd.read_csv(path)
    if "query_id" not in df.columns:
        raise ValueError("Prediction file must contain query_id.")
    label_column = "pred_label" if "pred_label" in df.columns else "label"
    if label_column not in df.columns:
        raise ValueError("Prediction file must contain pred_label or label.")
    rows = {}
    for row in df.itertuples():
        item = {"pred_label": normalize_label(getattr(row, label_column))}
        if hasattr(row, "margin"):
            item["llm_margin"] = float(row.margin)
        if hasattr(row, "confidence"):
            item["llm_confidence"] = float(row.confidence)
        if hasattr(row, "fallback_used"):
            item["fallback_used"] = bool(row.fallback_used)
        rows[str(row.query_id)] = item
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    runs = {
        "bm25": load_run(config.outputs.run_dir / f"{args.split}_bm25.csv"),
        "dense": load_run(config.outputs.run_dir / f"{args.split}_dense.csv"),
        "hybrid": load_run(config.outputs.run_dir / f"{args.split}_hybrid.csv"),
    }
    labels_df = pd.read_csv(config.outputs.run_dir / f"{args.split}_oracle_labels.csv")
    oracle_labels = dict(zip(labels_df["query_id"].astype(str), labels_df["oracle_label"].astype(str)))
    predictions = load_optional_predictions(args.predictions)

    rows = []
    for query_id, oracle_label in oracle_labels.items():
        bm25_hits = runs["bm25"].get(query_id, [])
        dense_hits = runs["dense"].get(query_id, [])
        hybrid_hits = runs["hybrid"].get(query_id, [])
        row = {
            "query_id": query_id,
            "oracle_label": oracle_label,
            "bm25_top1_score": bm25_hits[0][1] if bm25_hits else 0.0,
            "dense_top1_score": dense_hits[0][1] if dense_hits else 0.0,
            "hybrid_top1_score": hybrid_hits[0][1] if hybrid_hits else 0.0,
            "bm25_score_gap": score_gap(bm25_hits),
            "dense_score_gap": score_gap(dense_hits),
            "hybrid_score_gap": score_gap(hybrid_hits),
            "bm25_dense_overlap@10": overlap_ratio(bm25_hits, dense_hits, 10),
            "bm25_dense_overlap@100": overlap_ratio(bm25_hits, dense_hits, 100),
            "hybrid_bm25_overlap@10": overlap_ratio(hybrid_hits, bm25_hits, 10),
            "hybrid_dense_overlap@10": overlap_ratio(hybrid_hits, dense_hits, 10),
            "rrf_agreement@10": (
                overlap_ratio(hybrid_hits, bm25_hits, 10) + overlap_ratio(hybrid_hits, dense_hits, 10)
            )
            / 2,
            "top1_same_bm25_dense": bool(
                bm25_hits and dense_hits and bm25_hits[0][0] == dense_hits[0][0]
            ),
        }
        row.update(predictions.get(query_id, {}))
        rows.append(row)

    output = Path(args.output) if args.output else config.outputs.run_dir / f"{args.split}_retrieval_diagnostics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    grouped = defaultdict(dict)
    for key in ("oracle_label", "pred_label"):
        if key not in rows[0]:
            continue
        for label in sorted({str(row.get(key)) for row in rows}):
            subset = [row for row in rows if str(row.get(key)) == label]
            grouped[key][label] = {
                "count": len(subset),
                "avg_bm25_score_gap": sum(row["bm25_score_gap"] for row in subset) / len(subset),
                "avg_dense_score_gap": sum(row["dense_score_gap"] for row in subset) / len(subset),
                "avg_bm25_dense_overlap@10": sum(row["bm25_dense_overlap@10"] for row in subset) / len(subset),
                "avg_rrf_agreement@10": sum(row["rrf_agreement@10"] for row in subset) / len(subset),
            }

    summary = {
        "rows": len(rows),
        "oracle_distribution": dict(Counter(row["oracle_label"] for row in rows)),
        "prediction_distribution": dict(Counter(row["pred_label"] for row in rows if "pred_label" in row)),
        "groups": grouped,
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
