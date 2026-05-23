from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.oracle import create_oracle_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = {
        "bm25": load_run(config.outputs.run_dir / f"{args.split}_bm25.csv"),
        "dense": load_run(config.outputs.run_dir / f"{args.split}_dense.csv"),
        "hybrid": load_run(config.outputs.run_dir / f"{args.split}_hybrid.csv"),
    }
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    labels = create_oracle_labels(runs, qrels, tie_break_order=config.router.tie_break_order)
    output = config.outputs.run_dir / f"{args.split}_oracle_labels.csv"
    labels.to_csv(output, index=False)
    print(f"Saved oracle labels to {output}")


if __name__ == "__main__":
    main()
