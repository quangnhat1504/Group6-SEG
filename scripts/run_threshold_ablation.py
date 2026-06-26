"""Phase 3 selective-reranking threshold ablation + effectiveness-vs-cost analysis.

Reuses two already-computed runs to sweep uncertainty thresholds WITHOUT re-running
the cross-encoder:
  - <split>_hybrid.csv         -> base candidate ordering (no rerank)
  - <split>_always_rerank.csv  -> every query reranked (Hybrid top-k reranked)

For any selective policy, each query's result is either the base ordering (skip
rerank) or the reranked ordering (rerank). We therefore mix the two runs per
operating point and evaluate, which makes a full threshold sweep essentially free.

Outputs:
  - runs/<split>_threshold_ablation.csv     (every operating point, all policies)
  - reports/figures/phase3_pareto.png       (nDCG@10 vs rerank coverage)
  - reports/tables/table3_selective_rerank.{md,csv}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.fusion import top_doc_overlap
from seg_retrieval.io import load_qrels, load_queries, load_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.uncertainty import score_gap

RANDOM_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def build_selective_run(base, reranked, rerank_ids):
    """Per query: reranked ordering if in rerank_ids, else base ordering."""
    rerank_ids = set(rerank_ids)
    return {
        qid: (reranked.get(qid, base.get(qid, [])) if qid in rerank_ids else base.get(qid, []))
        for qid in base
    }


def operating_point(base, reranked, qrels, rerank_ids, latency_per_rerank, n_queries):
    run = build_selective_run(base, reranked, rerank_ids)
    metrics = evaluate_run(run, qrels)
    coverage = len(set(rerank_ids)) / n_queries if n_queries else 0.0
    metrics["rerank_coverage"] = coverage
    metrics["est_latency_ms_per_query"] = coverage * latency_per_rerank
    return metrics


def evaluate_subset(base, reranked, qrels, rerank_ids, subset_ids, latency_per_rerank):
    """Evaluate a selective policy on a subset of queries only (non-leaky eval)."""
    subset = set(subset_ids)
    sub_qrels = {q: labels for q, labels in qrels.items() if q in subset}
    run = build_selective_run(base, reranked, rerank_ids)
    sub_run = {q: hits for q, hits in run.items() if q in subset}
    metrics = evaluate_run(sub_run, sub_qrels)
    reranked_in_subset = len(set(rerank_ids) & subset)
    coverage = reranked_in_subset / len(subset) if subset else 0.0
    metrics["rerank_coverage"] = coverage
    metrics["est_latency_ms_per_query"] = coverage * latency_per_rerank
    return metrics


def select_threshold_on_calibration(signal, rule, thresholds, base, reranked, qrels, cal_ids, latency):
    """Pick the threshold that maximizes nDCG@10 on the calibration queries."""
    best_thr, best_ndcg = None, -1.0
    for thr in thresholds:
        rerank_ids = [qid for qid in cal_ids if rule(signal[qid], thr)]
        m = evaluate_subset(base, reranked, qrels, rerank_ids, cal_ids, latency)
        if m["ndcg@10"] > best_ndcg:
            best_ndcg, best_thr = m["ndcg@10"], thr
    return best_thr, best_ndcg


def sweep_signal(base, reranked, qrels, signals, rule, thresholds, latency, n_queries, policy):
    rows = []
    for thr in thresholds:
        rerank_ids = [qid for qid, val in signals.items() if rule(val, thr)]
        m = operating_point(base, reranked, qrels, rerank_ids, latency, n_queries)
        rows.append({"policy": policy, "threshold": float(thr), **m})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gain-recovery", type=float, default=0.90,
                        help="Headline selective point recovers >= this fraction of the Always-Rerank nDCG gain.")
    parser.add_argument("--calibration-fraction", type=float, default=0.5,
                        help="Fraction of queries used to tune the threshold (rest is held-out eval).")
    parser.add_argument("--seed", type=int, default=13, help="Calibration/eval split seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    base = load_run(run_dir / f"{args.split}_hybrid.csv")          # no-rerank candidate set
    reranked = load_run(run_dir / f"{args.split}_always_rerank.csv")  # everything reranked
    bm25 = load_run(run_dir / f"{args.split}_bm25.csv")
    dense = load_run(run_dir / f"{args.split}_dense.csv")

    query_ids = [q.query_id for q in queries]
    n = len(query_ids)

    # Per-query uncertainty signals (computed from runs, no labels used).
    gap_signal = {qid: score_gap(base.get(qid, [])) for qid in query_ids}
    disagree_signal = {
        qid: 1.0 - top_doc_overlap(bm25.get(qid, []), dense.get(qid, []), k=10) for qid in query_ids
    }

    # Per-query nDCG for base vs reranked -> oracle-selective skyline and analysis.
    base_ndcg = per_query_ndcg(base, qrels, 10)
    rerank_ndcg = per_query_ndcg(reranked, qrels, 10)
    gain = {qid: rerank_ndcg.get(qid, 0.0) - base_ndcg.get(qid, 0.0) for qid in query_ids}

    # Reference endpoints.
    latency = json.loads((run_dir / f"{args.split}_always_rerank_metrics.json").read_text())["latency_ms_per_query"]
    hybrid_metrics = evaluate_run(base, qrels)
    always_metrics = evaluate_run(reranked, qrels)
    hybrid_ndcg = hybrid_metrics["ndcg@10"]
    always_ndcg = always_metrics["ndcg@10"]
    total_gain = always_ndcg - hybrid_ndcg

    rows = []

    # Policy 1: rerank when score_gap < threshold (small gap = uncertain top rank).
    gap_thresholds = sorted(set(gap_signal.values()))
    gap_thresholds = [0.0] + gap_thresholds + [max(gap_thresholds) + 1e-6]
    rows += sweep_signal(base, reranked, qrels, gap_signal,
                         lambda v, t: v < t, gap_thresholds, latency, n, "score_gap")

    # Policy 2: rerank when disagreement >= threshold (high disagreement = uncertain).
    dis_thresholds = sorted(set(disagree_signal.values()))
    dis_thresholds = [-1e-6] + dis_thresholds + [1.0 + 1e-6]
    rows += sweep_signal(base, reranked, qrels, disagree_signal,
                         lambda v, t: v >= t, dis_thresholds, latency, n, "disagreement")

    # Policy 3: random selection (reference) - mean over seeds at each coverage budget.
    for frac in np.linspace(0.0, 1.0, 21):
        k = round(frac * n)
        per_seed = []
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(query_ids, size=k, replace=False) if k else []
            per_seed.append(operating_point(base, reranked, qrels, chosen, latency, n))
        rows.append({
            "policy": "random", "threshold": float(frac),
            "ndcg@10": float(np.mean([m["ndcg@10"] for m in per_seed])),
            "ndcg@10_std": float(np.std([m["ndcg@10"] for m in per_seed])),
            "recall@10": float(np.mean([m["recall@10"] for m in per_seed])),
            "recall@100": float(np.mean([m["recall@100"] for m in per_seed])),
            "mrr@10": float(np.mean([m["mrr@10"] for m in per_seed])),
            "rerank_coverage": k / n if n else 0.0,
            "est_latency_ms_per_query": (k / n if n else 0.0) * latency,
        })

    # Policy 4: oracle-selective skyline - rerank the queries that gain most first.
    ranked_by_gain = sorted(query_ids, key=lambda q: gain[q], reverse=True)
    for k in range(0, n + 1):
        chosen = ranked_by_gain[:k]
        m = operating_point(base, reranked, qrels, chosen, latency, n)
        rows.append({"policy": "oracle_selective", "threshold": float(k), **m})

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"{args.split}_threshold_ablation.csv", index=False)

    # --- Signal quality: area under nDCG-vs-coverage curve (higher = better signal) ---
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

    def auc(policy):
        sub = df[df.policy == policy].sort_values("rerank_coverage")
        return float(trapezoid(sub["ndcg@10"], sub["rerank_coverage"]))

    auc_summary = {p: auc(p) for p in ["score_gap", "disagreement", "random", "oracle_selective"]}

    # --- Headline selective operating point: smallest coverage on the BEST signal
    #     curve that recovers >= gain_recovery of the Always-Rerank nDCG gain. ---
    target_ndcg = hybrid_ndcg + args.gain_recovery * total_gain
    best_signal = max(["score_gap", "disagreement"], key=lambda p: auc(p))
    cand = df[(df.policy == best_signal) & (df["ndcg@10"] >= target_ndcg)]
    if len(cand):
        headline = cand.sort_values("rerank_coverage").iloc[0].to_dict()
    else:  # fall back to the max-nDCG point of the best signal
        headline = df[df.policy == best_signal].sort_values("ndcg@10").iloc[-1].to_dict()

    # Config-default operating point (score_gap<g OR disagreement>d) for reference.
    g = config.rerank.score_gap_threshold
    d = config.rerank.disagreement_threshold
    default_ids = [q for q in query_ids if gap_signal[q] < g or disagree_signal[q] > d]
    default_point = operating_point(base, reranked, qrels, default_ids, latency, n)

    # --- Non-leaky operating point: tune threshold on calibration, report on eval ---
    rng = np.random.default_rng(args.seed)
    shuffled = list(query_ids)
    rng.shuffle(shuffled)
    n_cal = int(round(args.calibration_fraction * n))
    cal_ids, eval_ids = shuffled[:n_cal], shuffled[n_cal:]
    signal_map = {"score_gap": (gap_signal, lambda v, t: v < t, gap_thresholds),
                  "disagreement": (disagree_signal, lambda v, t: v >= t, dis_thresholds)}
    sig, rule, thrs = signal_map[best_signal]
    tuned_thr, tuned_cal_ndcg = select_threshold_on_calibration(
        sig, rule, thrs, base, reranked, qrels, cal_ids, latency)
    eval_rerank_ids = [q for q in eval_ids if rule(sig[q], tuned_thr)]
    heldout_point = evaluate_subset(base, reranked, qrels, eval_rerank_ids, eval_ids, latency)
    # Reference endpoints on the SAME eval subset for a fair comparison.
    eval_hybrid = evaluate_subset(base, reranked, qrels, [], eval_ids, latency)
    eval_always = evaluate_subset(base, reranked, qrels, eval_ids, eval_ids, latency)

    # ---------------- Pareto plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for policy, label, style in [
        ("oracle_selective", "Oracle-selective (skyline)", {"color": "#444", "ls": "--"}),
        ("score_gap", "Selective: score-gap", {"color": "#1f77b4", "marker": "o", "ms": 3}),
        ("disagreement", "Selective: disagreement", {"color": "#2ca02c", "marker": "s", "ms": 3}),
    ]:
        sub = df[df.policy == policy].sort_values("rerank_coverage")
        ax.plot(sub["rerank_coverage"], sub["ndcg@10"], label=label, **style)
    rnd = df[df.policy == "random"].sort_values("rerank_coverage")
    ax.plot(rnd["rerank_coverage"], rnd["ndcg@10"], color="#999", ls=":", label="Random (mean of 10 seeds)")
    ax.fill_between(rnd["rerank_coverage"], rnd["ndcg@10"] - rnd["ndcg@10_std"],
                    rnd["ndcg@10"] + rnd["ndcg@10_std"], color="#999", alpha=0.15)
    ax.axhline(hybrid_ndcg, color="#d62728", ls="-.", lw=1, label=f"Hybrid base ({hybrid_ndcg:.4f})")
    ax.axhline(always_ndcg, color="#9467bd", ls="-.", lw=1, label=f"Always-Rerank ({always_ndcg:.4f})")
    ax.scatter([headline["rerank_coverage"]], [headline["ndcg@10"]], color="#1f77b4",
               s=90, zorder=5, edgecolor="k",
               label=f"Headline: {headline['rerank_coverage']*100:.0f}% cov -> {headline['ndcg@10']:.4f}")
    ax.set_xlabel("Rerank coverage (fraction of queries reranked)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Phase 3 selective reranking: effectiveness vs cost (SciFact test)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "phase3_pareto.png", dpi=150)

    # ---------------- Table 3 ----------------
    bm = json.loads((run_dir / f"{args.split}_base_metrics.json").read_text())
    table_rows = [
        ("BM25", bm["bm25"], 0.0, 0.0),
        ("Dense / SciNCL", bm["dense"], 0.0, 0.0),
        ("Hybrid RRF", hybrid_metrics, 0.0, 0.0),
        ("Selective rerank (score-gap, headline)", headline,
         headline["rerank_coverage"], headline["est_latency_ms_per_query"]),
        ("Selective rerank (config default OR-rule)", default_point,
         default_point["rerank_coverage"], default_point["est_latency_ms_per_query"]),
        ("Always-Rerank", always_metrics, 1.0, latency),
    ]
    md = ["| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Rerank Coverage | Est. ms/query |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for name, m, cov, lat in table_rows:
        md.append(f"| {name} | {m['ndcg@10']:.4f} | {m['recall@10']:.4f} | "
                  f"{m['recall@100']:.4f} | {m['mrr@10']:.4f} | {cov:.2f} | {lat:.1f} |")
        csv_rows.append({"method": name, "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"],
                         "recall@100": m["recall@100"], "mrr@10": m["mrr@10"],
                         "rerank_coverage": cov, "est_latency_ms_per_query": lat})
    (reports / "tables" / "table3_selective_rerank.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(reports / "tables" / "table3_selective_rerank.csv", index=False)

    # Non-leaky held-out table (threshold tuned on calibration, evaluated on disjoint eval).
    md2 = [f"Threshold tuned on {len(cal_ids)} calibration queries (seed={args.seed}), "
           f"evaluated on {len(eval_ids)} disjoint queries. Signal: {best_signal}, tuned threshold={tuned_thr:.4g}.",
           "",
           "| Method (eval subset) | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Rerank Coverage | Est. ms/query |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m, cov in [
        ("Hybrid RRF (no rerank)", eval_hybrid, 0.0),
        ("Selective rerank (held-out tuned)", heldout_point, heldout_point["rerank_coverage"]),
        ("Always-Rerank", eval_always, 1.0),
    ]:
        md2.append(f"| {name} | {m['ndcg@10']:.4f} | {m['recall@10']:.4f} | {m['recall@100']:.4f} | "
                   f"{m['mrr@10']:.4f} | {cov:.2f} | {cov * latency:.1f} |")
    (reports / "tables" / "table3b_selective_heldout.md").write_text("\n".join(md2) + "\n", encoding="utf-8")

    # ---------------- Console summary ----------------
    print("=" * 70)
    print("PHASE 3 THRESHOLD ABLATION — SUMMARY")
    print("=" * 70)
    print(f"Hybrid base nDCG@10      : {hybrid_ndcg:.4f}")
    print(f"Always-Rerank nDCG@10    : {always_ndcg:.4f}  (gain +{total_gain:.4f}, {latency:.1f} ms/q)")
    print(f"Best uncertainty signal  : {best_signal}  (AUC {auc_summary[best_signal]:.4f})")
    print(f"Signal AUC (nDCG vs cov) : " + ", ".join(f"{p}={v:.4f}" for p, v in auc_summary.items()))
    print(f"\nHeadline selective point (>= {args.gain_recovery:.0%} of gain via {best_signal}):")
    print(f"  coverage={headline['rerank_coverage']:.2%}  nDCG@10={headline['ndcg@10']:.4f}  "
          f"recall@10={headline['recall@10']:.4f}  mrr@10={headline['mrr@10']:.4f}  "
          f"~{headline['est_latency_ms_per_query']:.1f} ms/q")
    print(f"  -> recovers {(headline['ndcg@10']-hybrid_ndcg)/total_gain:.1%} of the gain at "
          f"{headline['rerank_coverage']:.0%} of the rerank cost")
    print(f"\nConfig-default OR-rule point (full test):")
    print(f"  coverage={default_point['rerank_coverage']:.2%}  nDCG@10={default_point['ndcg@10']:.4f}")
    print(f"\nNon-leaky held-out ({best_signal}, tune on {len(cal_ids)} cal, eval on {len(eval_ids)}):")
    print(f"  tuned threshold={tuned_thr:.4g} (cal nDCG@10={tuned_cal_ndcg:.4f})")
    print(f"  EVAL  hybrid={eval_hybrid['ndcg@10']:.4f}  selective={heldout_point['ndcg@10']:.4f}  "
          f"always={eval_always['ndcg@10']:.4f}  | coverage={heldout_point['rerank_coverage']:.2%}")
    print(f"\nWrote:")
    print(f"  {run_dir / (args.split + '_threshold_ablation.csv')}")
    print(f"  {reports / 'figures' / 'phase3_pareto.png'}")
    print(f"  {reports / 'tables' / 'table3_selective_rerank.md'}")


if __name__ == "__main__":
    main()
