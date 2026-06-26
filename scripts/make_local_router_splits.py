"""Carve the SciFact train split into disjoint trainfit + dev sub-splits.

Motivation: the earlier LLM calibration had no clean dev set, so it split the TEST
prediction file against itself (calibration vs eval on the same 300 test queries).
To get a genuine train/dev/test protocol fully on local hardware, we instead:

  * fine-tune the QLoRA router only on `trainfit`,
  * tune calibration on `dev` (held out from training),
  * report final numbers on the untouched `test` split.

This script materialises `trainfit` and `dev` as first-class splits by subsetting
every per-split artefact the downstream scripts expect, using the SAME file-naming
convention (``<split>_queries.jsonl``, ``<split>_qrels.csv``,
``<split>_oracle_labels.csv`` and the ``<split>_{bm25,dense,hybrid}.csv`` runs).
The dev split is stratified by oracle label so all three routes are represented.

Inputs (must already exist for the source split, default ``train``):
  data/scifact/train_queries.jsonl, train_qrels.csv
  runs/scifact/train_oracle_labels.csv
  runs/scifact/train_{bm25,dense,hybrid}.csv

Outputs (for split in {trainfit, dev}):
  data/scifact/<split>_queries.jsonl, <split>_qrels.csv
  runs/scifact/<split>_oracle_labels.csv, <split>_{bm25,dense,hybrid}.csv
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_queries, save_queries


def stratified_dev_ids(
    oracle: pd.DataFrame, dev_fraction: float, seed: int
) -> set[str]:
    """Pick a dev set stratified by oracle_label (at least 1 per present label)."""
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in oracle.itertuples():
        by_label[str(row.oracle_label)].append(str(row.query_id))

    dev_ids: set[str] = set()
    for label, ids in by_label.items():
        rng.shuffle(ids)
        k = max(1, round(len(ids) * dev_fraction))
        k = min(k, len(ids) - 1) if len(ids) > 1 else len(ids)
        dev_ids.update(ids[:k])
    return dev_ids


def subset_run_csv(src: Path, dst: Path, keep_ids: set[str]) -> None:
    df = pd.read_csv(src)
    df["query_id"] = df["query_id"].astype(str)
    df[df["query_id"].isin(keep_ids)].to_csv(dst, index=False)


def subset_qrels_csv(src: Path, dst: Path, keep_ids: set[str]) -> None:
    df = pd.read_csv(src)
    df["query_id"] = df["query_id"].astype(str)
    df[df["query_id"].isin(keep_ids)].to_csv(dst, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--trainfit-name", default="trainfit")
    parser.add_argument("--dev-name", default="dev")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.dataset.data_dir
    run_dir = config.outputs.run_dir
    src = args.source_split

    oracle = pd.read_csv(run_dir / f"{src}_oracle_labels.csv")
    oracle["query_id"] = oracle["query_id"].astype(str)
    all_ids = set(oracle["query_id"])

    dev_ids = stratified_dev_ids(oracle, args.dev_fraction, args.seed)
    trainfit_ids = all_ids - dev_ids

    queries = load_queries(data_dir / f"{src}_queries.jsonl")

    for name, keep in [(args.trainfit_name, trainfit_ids), (args.dev_name, dev_ids)]:
        # queries
        save_queries(
            data_dir / f"{name}_queries.jsonl",
            [q for q in queries if q.query_id in keep],
        )
        # qrels
        subset_qrels_csv(data_dir / f"{src}_qrels.csv", data_dir / f"{name}_qrels.csv", keep)
        # oracle labels
        oracle[oracle["query_id"].isin(keep)].to_csv(
            run_dir / f"{name}_oracle_labels.csv", index=False
        )
        # runs
        for route in ("bm25", "dense", "hybrid"):
            subset_run_csv(
                run_dir / f"{src}_{route}.csv", run_dir / f"{name}_{route}.csv", keep
            )

    print(
        f"Source '{src}': {len(all_ids)} queries -> "
        f"'{args.trainfit_name}'={len(trainfit_ids)}, '{args.dev_name}'={len(dev_ids)}"
    )
    label_counts = (
        oracle.assign(split=oracle["query_id"].map(lambda q: args.dev_name if q in dev_ids else args.trainfit_name))
        .groupby(["split", "oracle_label"])
        .size()
        .unstack(fill_value=0)
    )
    print(label_counts.to_string())


if __name__ == "__main__":
    main()
