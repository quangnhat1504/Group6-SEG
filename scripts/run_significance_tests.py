"""Paired bootstrap significance tests over queries.

For each (system, baseline) pair we compute per-query nDCG@10 and MRR@10 on the
shared query set, then estimate the mean paired difference with a 95% percentile
bootstrap confidence interval and a two-sided bootstrap p-value. This replaces the
point-estimate-only tables with proper paired uncertainty.

Outputs:
  - runs/<split>_significance_tests.csv
  - reports/tables/table_significance.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.metrics import per_query_ndcg
from seg_retrieval.types import Qrels, Run

N_BOOT = 10000
SEED = 13


def per_query_mrr(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    out: dict[str, float] = {}
    for query_id, labels in qrels.items():
        rr = 0.0
        for rank, (doc_id, _) in enumerate(run.get(query_id, [])[:k], start=1):
            if labels.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
        out[query_id] = rr
    return out


def paired_bootstrap(sys_vals: np.ndarray, base_vals: np.ndarray, rng: np.random.Generator):
    diff = sys_vals - base_vals
    n = len(diff)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        means[b] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the resampled mean crosses 0.
    frac_le = float(np.mean(means <= 0.0))
    frac_ge = float(np.mean(means >= 0.0))
    p = 2.0 * min(frac_le, frac_ge)
    return float(diff.mean()), float(lo), float(hi), min(p, 1.0)


def compare(sys_run: Run, base_run: Run, qrels: Qrels, rng: np.random.Generator) -> dict:
    shared = [q for q in qrels if q in sys_run and q in base_run]
    sub_qrels = {q: qrels[q] for q in shared}
    out = {"n_queries": len(shared)}
    for metric, fn in (("ndcg@10", per_query_ndcg), ("mrr@10", per_query_mrr)):
        sysd = fn(sys_run, sub_qrels, 10)
        based = fn(base_run, sub_qrels, 10)
        s = np.array([sysd[q] for q in shared])
        b = np.array([based[q] for q in shared])
        mean_diff, lo, hi, p = paired_bootstrap(s, b, rng)
        out[f"{metric}_sys"] = float(s.mean())
        out[f"{metric}_base"] = float(b.mean())
        out[f"{metric}_diff"] = mean_diff
        out[f"{metric}_ci_lo"] = lo
        out[f"{metric}_ci_hi"] = hi
        out[f"{metric}_p"] = p
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")

    def R(name: str) -> Run:
        return load_run(run_dir / name)

    pairs = [
        ("Hybrid RRF vs BM25", f"{args.split}_hybrid.csv", f"{args.split}_bm25.csv"),
        ("Always-Rerank vs Hybrid RRF", f"{args.split}_always_rerank.csv", f"{args.split}_hybrid.csv"),
        ("Oracle Router vs Hybrid RRF", f"{args.split}_oracle_router.csv", f"{args.split}_hybrid.csv"),
    ]
    # Optional: calibrated LLM held-out route vs BM25 on its own (150) query subset.
    cal_routed = run_dir / f"{args.split}_test_llm_router_predictions_2_calibrated_heldout_routed.csv"
    if cal_routed.exists():
        pairs.append(("Calibrated LLM (held-out) vs BM25",
                      cal_routed.name, f"{args.split}_bm25.csv"))

    rng = np.random.default_rng(SEED)
    rows = []
    for label, sys_name, base_name in pairs:
        res = compare(R(sys_name), R(base_name), qrels, rng)
        res = {"comparison": label, **res}
        rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"{args.split}_significance_tests.csv", index=False)

    md = [
        f"Paired bootstrap significance tests on SciFact {args.split} "
        f"({N_BOOT} resamples, seed={SEED}). CI = 95% percentile interval of the mean paired "
        f"difference; p = two-sided bootstrap p-value. Bold = CI excludes 0.",
        "",
        "| Comparison | n | Metric | System | Baseline | Mean diff | 95% CI | p |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        for metric in ("ndcg@10", "mrr@10"):
            sig = r[f"{metric}_ci_lo"] > 0 or r[f"{metric}_ci_hi"] < 0
            diff_str = f"**{r[f'{metric}_diff']:+.4f}**" if sig else f"{r[f'{metric}_diff']:+.4f}"
            ci = f"[{r[f'{metric}_ci_lo']:+.4f}, {r[f'{metric}_ci_hi']:+.4f}]"
            md.append(
                f"| {r['comparison']} | {r['n_queries']} | {metric} | {r[f'{metric}_sys']:.4f} | "
                f"{r[f'{metric}_base']:.4f} | {diff_str} | {ci} | {r[f'{metric}_p']:.4f} |"
            )
    (reports / "tables" / "table_significance.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
