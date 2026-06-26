from __future__ import annotations

import argparse
from collections import defaultdict
import random

import _bootstrap  # noqa: F401

import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_queries, write_jsonl


SYSTEM_PROMPT = (
    "You are a query router for a scientific paper search engine. "
    "Choose exactly one retrieval route from: bm25, dense, hybrid. "
    "Your whole answer must be only the route label."
)


def build_prompt(query: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Academic query:\n{query}\n\n"
        "Return only one label: bm25, dense, or hybrid."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None)
    parser.add_argument("--balance", action="store_true", help="Upsample minority labels to the largest label count.")
    parser.add_argument(
        "--balance-mode",
        choices=("upsample", "undersample"),
        default="upsample",
        help="Sampling strategy used when --balance is set.",
    )
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    config = load_config(args.config)
    output = args.output or config.outputs.run_dir / f"{args.split}_llm_router_data.jsonl"
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    query_by_id = {query.query_id: query.text for query in queries}
    labels = pd.read_csv(config.outputs.run_dir / f"{args.split}_oracle_labels.csv")

    rows = []
    for row in labels.itertuples():
        query_id = str(row.query_id)
        if query_id not in query_by_id:
            continue
        label = str(row.oracle_label).strip().lower()
        rows.append(
            {
                "query_id": query_id,
                "query": query_by_id[query_id],
                "prompt": build_prompt(query_by_id[query_id]),
                "label": label,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Academic query:\n{query_by_id[query_id]}\n\nRoute:"},
                    {"role": "assistant", "content": label},
                ],
            }
        )

    if args.balance and rows:
        rng = random.Random(args.seed)
        by_label: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        target_count = (
            max(len(group) for group in by_label.values())
            if args.balance_mode == "upsample"
            else min(len(group) for group in by_label.values())
        )
        balanced_rows = []
        for label, group in by_label.items():
            if args.balance_mode == "upsample":
                balanced_rows.extend(group)
                balanced_rows.extend(rng.choices(group, k=target_count - len(group)))
            else:
                balanced_rows.extend(rng.sample(group, k=target_count))
        rng.shuffle(balanced_rows)
        rows = balanced_rows

    write_jsonl(output, rows)
    print(f"Saved {len(rows)} LLM router examples to {output}")


if __name__ == "__main__":
    main()
