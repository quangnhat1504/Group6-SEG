"""Paired bootstrap significance test: conformal selective reranking vs Always-Rerank.

Compares the QPP-conformal selective reranking strategy (hybrid_max, alpha=0.02)
against the Always-Rerank baseline on the evaluation subset (150 queries).

The test uses paired bootstrap resampling (10,000 resamples) to estimate
confidence intervals and p-values for both nDCG@10 and MRR@10.

Outputs:
  - runs/scifact/test_significance_conformal.csv
  - reports/tables/table_significance_conformal.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.types import Qrels, Run

N_BOOT = 10000
SEED = 13


def per_query_mrr(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    """Compute per-query MRR@k."""
    out: dict[str, float] = {}
    for query_id, labels in qrels.items():
        rr = 0.0
        for rank, (doc_id, _) in enumerate(run.get(query_id, [])[:k], start=1):
            if labels.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
        out[query_id] = rr
    return out


def paired_bootstrap(
    sys_vals: np.ndarray, base_vals: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float, float]:
    """Paired bootstrap test returning (mean_diff, ci_lo, ci_hi, p_value)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance test: conformal selective vs Always-Rerank"
    )
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    data_dir = config.dataset.data_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    # ---- Load QPP features (per-query nDCG values and hybrid_max signal) ----
    feats = pd.read_csv(run_dir / f"{args.split}_qpp_features.csv")
    feats["query_id"] = feats["query_id"].astype(str)
    ids = feats["query_id"].tolist()

    base_ndcg = dict(zip(feats["query_id"], feats["base_ndcg"]))
    rerank_ndcg = dict(zip(feats["query_id"], feats["rerank_ndcg"]))
    hybrid_max = dict(zip(feats["query_id"], feats["hybrid_max"]))

    # ---- Reconstruct eval subset (matches run_conformal_rerank.py exactly) ----
    rng = np.random.default_rng(SEED)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_cal = int(round(0.5 * len(ids)))
    eval_ids = shuffled[n_cal:]  # second half is eval

    # ---- Extract lambda at alpha=0.02, signal=hybrid_max ----
    conformal_df = pd.read_csv(run_dir / f"{args.split}_conformal_results.csv")
    mask = (conformal_df["signal"] == "hybrid_max") & (
        np.isclose(conformal_df["alpha"], 0.02)
    )
    lam = conformal_df.loc[mask, "lambda"].iloc[0]

    # ---- Build per-query nDCG arrays for the comparison ----
    # Conformal selective: queries with hybrid_max <= lambda get rerank_ndcg, others get base_ndcg
    conformal_ndcg = np.array([
        rerank_ndcg[q] if hybrid_max[q] <= lam else base_ndcg[q]
        for q in eval_ids
    ])
    # Always-Rerank: all queries get rerank_ndcg
    always_rerank_ndcg = np.array([rerank_ndcg[q] for q in eval_ids])

    # ---- Build per-query MRR arrays ----
    # Load the actual runs to compute MRR from ranked lists
    qrels = load_qrels(data_dir / f"{args.split}_qrels.csv")
    hybrid_run = load_run(run_dir / f"{args.split}_hybrid.csv")
    rerank_run = load_run(run_dir / f"{args.split}_always_rerank.csv")

    # Compute per-query MRR for both runs
    hybrid_mrr = per_query_mrr(hybrid_run, qrels, k=10)
    rerank_mrr = per_query_mrr(rerank_run, qrels, k=10)

    # Conformal selective MRR: same decision logic as nDCG
    conformal_mrr = np.array([
        rerank_mrr.get(q, 0.0) if hybrid_max[q] <= lam else hybrid_mrr.get(q, 0.0)
        for q in eval_ids
    ])
    # Always-Rerank MRR
    always_rerank_mrr = np.array([rerank_mrr.get(q, 0.0) for q in eval_ids])

    # ---- Run paired bootstrap tests ----
    rng_boot = np.random.default_rng(SEED)

    rows = []
    for metric_name, sys_vals, base_vals in [
        ("ndcg@10", conformal_ndcg, always_rerank_ndcg),
        ("mrr@10", conformal_mrr, always_rerank_mrr),
    ]:
        mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(sys_vals, base_vals, rng_boot)
        significant = ci_lo > 0 or ci_hi < 0
        rows.append({
            "comparison": "Conformal Selective vs Always-Rerank",
            "metric": metric_name,
            "n_queries": len(eval_ids),
            "system_mean": float(sys_vals.mean()),
            "baseline_mean": float(base_vals.mean()),
            "mean_diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": p_value,
            "significant": significant,
        })

    # ---- Output CSV ----
    df_out = pd.DataFrame(rows)
    out_csv = run_dir / f"{args.split}_significance_conformal.csv"
    df_out.to_csv(out_csv, index=False)

    # ---- Output Markdown table ----
    md_lines = [
        f"Paired bootstrap significance test: Conformal Selective Reranking (hybrid_max, alpha=0.02) "
        f"vs Always-Rerank on the evaluation subset ({len(eval_ids)} queries, "
        f"{N_BOOT} resamples, seed={SEED}).",
        "",
        f"Lambda threshold: {lam:.10f}",
        f"Rerank coverage: {sum(1 for q in eval_ids if hybrid_max[q] <= lam) / len(eval_ids):.2%}",
        "",
        "| Metric | System | Baseline | Mean diff | 95% CI | p | Significant |",
        "|---|---:|---:|---:|---|---:|:--:|",
    ]
    for r in rows:
        sig_mark = "**yes**" if r["significant"] else "no"
        diff_str = f"**{r['mean_diff']:+.4f}**" if r["significant"] else f"{r['mean_diff']:+.4f}"
        ci_str = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        md_lines.append(
            f"| {r['metric']} | {r['system_mean']:.4f} | {r['baseline_mean']:.4f} | "
            f"{diff_str} | {ci_str} | {r['p_value']:.4f} | {sig_mark} |"
        )

    md_path = reports / "tables" / "table_significance_conformal.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # ---- Console summary ----
    print("=" * 70)
    print("CONFORMAL SELECTIVE vs ALWAYS-RERANK — SIGNIFICANCE TEST")
    print("=" * 70)
    print(f"Eval subset: {len(eval_ids)} queries (seed={SEED}, 50% split)")
    print(f"Lambda (hybrid_max, alpha=0.02): {lam:.10f}")
    print(f"Rerank coverage: {sum(1 for q in eval_ids if hybrid_max[q] <= lam) / len(eval_ids):.2%}")
    print()
    for r in rows:
        sig_str = "SIGNIFICANT" if r["significant"] else "not significant"
        print(f"  {r['metric']:8s}: system={r['system_mean']:.4f}  baseline={r['baseline_mean']:.4f}  "
              f"diff={r['mean_diff']:+.4f}  CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
              f"p={r['p_value']:.4f}  [{sig_str}]")
    print()
    print(f"Wrote: {out_csv}")
    print(f"       {md_path}")


if __name__ == "__main__":
    main()
