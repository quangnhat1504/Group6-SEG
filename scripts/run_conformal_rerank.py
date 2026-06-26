"""Direction 1: conformal risk-controlled selective reranking.

Uses Conformal Risk Control (Angelopoulos et al., 2024) to pick the rerank-trigger
threshold with a guarantee on the expected nDCG shortfall versus Always-Rerank.

Setup. A query is reranked when its signal s(q) <= lambda (low signal = weak base
result = reranking likely helps). The per-query loss is the nDCG forfeited by NOT
reranking a query:
    l(q) = max(0, rerank_ndcg(q) - base_ndcg(q))      if q is skipped
         = 0                                          if q is reranked
l(q, lambda) is monotone non-increasing in lambda (reranking more never raises the
loss), so CRC applies. CRC chooses the smallest-coverage lambda such that
    (n * Rhat(lambda) + B) / (n + 1) <= alpha
on the calibration split, which guarantees E[risk] <= alpha on the held-out split,
where risk = mean nDCG shortfall vs Always-Rerank over skipped queries and B is the
loss upper bound (B = 1).

We run CRC for two trigger signals to show the Direction-2 payoff: a better QPP
signal (hybrid_max) needs less rerank coverage than the Phase-3 score-gap heuristic
to honour the same risk level.

Outputs:
  - runs/<split>_conformal_results.csv               (per alpha, per signal)
  - reports/figures/phase3_conformal_risk_coverage.png
  - reports/tables/table4_conformal.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config

SIGNALS = {
    # name -> (feature column, human label). Rerank when feature <= lambda.
    "hybrid_max": ("hybrid_max", "QPP: hybrid max score (Direction 2)"),
    "score_gap": ("hybrid_gap", "Phase-3 heuristic: score gap"),
}
LOSS_BOUND = 1.0  # nDCG difference is in [0, 1]


def crc_threshold(signal, loss, cal_ids, alpha, B=LOSS_BOUND):
    """Smallest-coverage lambda on calibration with (n*Rhat + B)/(n+1) <= alpha."""
    n = len(cal_ids)
    candidates = [float("-inf")] + sorted({signal[q] for q in cal_ids})
    for lam in candidates:  # ascending lambda -> coverage increases, Rhat decreases
        skipped = [q for q in cal_ids if signal[q] > lam]
        rhat = sum(loss[q] for q in skipped) / n
        if (n * rhat + B) / (n + 1) <= alpha:
            return lam
    return candidates[-1]  # rerank all


def evaluate(signal, loss, base_ndcg, rerank_ndcg, ids, lam):
    skipped = [q for q in ids if signal[q] > lam]
    reranked = [q for q in ids if signal[q] <= lam]
    n = len(ids)
    risk = sum(loss[q] for q in skipped) / n if n else 0.0  # mean shortfall vs always-rerank
    ndcg = (sum(base_ndcg[q] for q in skipped) + sum(rerank_ndcg[q] for q in reranked)) / n if n else 0.0
    coverage = len(reranked) / n if n else 0.0
    return {"coverage": coverage, "ndcg": ndcg, "risk": risk}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=13, help="Calibration/eval split seed.")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--headline-alpha", type=float, default=0.02)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(run_dir / f"{args.split}_qpp_features.csv")
    feats["query_id"] = feats["query_id"].astype(str)
    ids = feats["query_id"].tolist()
    base_ndcg = dict(zip(feats["query_id"], feats["base_ndcg"]))
    rerank_ndcg = dict(zip(feats["query_id"], feats["rerank_ndcg"]))
    loss = {q: max(0.0, rerank_ndcg[q] - base_ndcg[q]) for q in ids}

    latency = pd.read_json(run_dir / f"{args.split}_always_rerank_metrics.json", typ="series")["latency_ms_per_query"]

    # Calibration / evaluation split (same protocol as Phase 2/3).
    rng = np.random.default_rng(args.seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_cal = int(round(args.calibration_fraction * len(ids)))
    cal_ids, eval_ids = shuffled[:n_cal], shuffled[n_cal:]

    always_eval = sum(rerank_ndcg[q] for q in eval_ids) / len(eval_ids)
    hybrid_eval = sum(base_ndcg[q] for q in eval_ids) / len(eval_ids)

    signal_maps = {name: dict(zip(feats["query_id"], feats[col])) for name, (col, _) in SIGNALS.items()}
    alphas = np.round(np.arange(0.005, 0.0501, 0.0025), 4)

    # Single fixed split (seed) for the reported headline table (matches Phase 2/3 protocol).
    rows = []
    for name in SIGNALS:
        signal = signal_maps[name]
        for alpha in alphas:
            lam = crc_threshold(signal, loss, cal_ids, float(alpha))
            ev = evaluate(signal, loss, base_ndcg, rerank_ndcg, eval_ids, lam)
            rows.append({
                "signal": name, "alpha": float(alpha), "lambda": lam,
                "eval_coverage": ev["coverage"], "eval_ndcg": ev["ndcg"],
                "eval_risk": ev["risk"], "guarantee_ok": ev["risk"] <= alpha,
                "est_latency_ms_per_query": ev["coverage"] * latency,
            })
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"{args.split}_conformal_results.csv", index=False)

    # Multi-seed validation: CRC guarantees E[risk] <= alpha in expectation over the
    # calibration draw, so we average realized eval risk/coverage over many splits.
    n_seeds = 200
    agg = {(name, float(a)): {"risk": [], "cov": [], "ndcg": []} for name in SIGNALS for a in alphas}
    for seed in range(n_seeds):
        rng_s = np.random.default_rng(1000 + seed)
        sh = list(ids)
        rng_s.shuffle(sh)
        c_ids, e_ids = sh[:n_cal], sh[n_cal:]
        for name in SIGNALS:
            signal = signal_maps[name]
            for a in alphas:
                lam = crc_threshold(signal, loss, c_ids, float(a))
                ev = evaluate(signal, loss, base_ndcg, rerank_ndcg, e_ids, lam)
                agg[(name, float(a))]["risk"].append(ev["risk"])
                agg[(name, float(a))]["cov"].append(ev["coverage"])
                agg[(name, float(a))]["ndcg"].append(ev["ndcg"])
    mean_rows = []
    for (name, a), d in agg.items():
        mean_rows.append({
            "signal": name, "alpha": a,
            "mean_eval_risk": float(np.mean(d["risk"])),
            "mean_eval_coverage": float(np.mean(d["cov"])),
            "mean_eval_ndcg": float(np.mean(d["ndcg"])),
            "guarantee_ok_mean": float(np.mean(d["risk"])) <= a,
        })
    mean_df = pd.DataFrame(mean_rows).sort_values(["signal", "alpha"])
    mean_df.to_csv(run_dir / f"{args.split}_conformal_results_mean.csv", index=False)

    # ---------------- Risk-coverage / guarantee figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"hybrid_max": "#1f77b4", "score_gap": "#2ca02c"}
    for name in SIGNALS:
        sub = mean_df[mean_df.signal == name].sort_values("alpha")
        ax1.plot(sub["alpha"], sub["mean_eval_coverage"], marker="o", ms=4,
                 color=colors[name], label=SIGNALS[name][1])
        ax2.plot(sub["alpha"], sub["mean_eval_risk"], marker="o", ms=4, color=colors[name], label=name)
    ax1.set_xlabel("Risk budget alpha (allowed nDCG shortfall vs Always-Rerank)")
    ax1.set_ylabel("Mean rerank coverage required")
    ax1.set_title("Cost to honour the guarantee\n(lower = better trigger signal)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot([alphas.min(), alphas.max()], [alphas.min(), alphas.max()],
             ls="--", color="#888", label="risk = alpha (budget)")
    ax2.set_xlabel("Risk budget alpha")
    ax2.set_ylabel("Mean realized risk (200 splits)")
    ax2.set_title("Guarantee check: E[risk] <= alpha")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle("Conformal risk-controlled selective reranking (SciFact, mean of 200 splits)", fontsize=12)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "phase3_conformal_risk_coverage.png", dpi=150)

    # ---------------- Headline table at chosen alpha ----------------
    a = args.headline_alpha
    md = [f"Conformal risk control at alpha = {a} (expected nDCG shortfall vs Always-Rerank), "
          f"delta via CRC expectation bound. Threshold tuned on {len(cal_ids)} calibration queries "
          f"(seed={args.seed}), evaluated on {len(eval_ids)} disjoint queries.",
          "",
          f"Reference: Hybrid (no rerank) nDCG@10 = {hybrid_eval:.4f}; "
          f"Always-Rerank nDCG@10 = {always_eval:.4f}.",
          "",
          "| Trigger signal | Rerank Coverage | nDCG@10 | Realized risk | Guarantee (<= alpha) | Est. ms/query |",
          "|---|---:|---:|---:|:--:|---:|"]
    headline = {}
    for name in SIGNALS:
        sub = df[(df.signal == name) & (np.isclose(df.alpha, a))]
        if not len(sub):
            continue
        r = sub.iloc[0]
        headline[name] = r
        md.append(f"| {SIGNALS[name][1]} | {r['eval_coverage']:.2f} | {r['eval_ndcg']:.4f} | "
                  f"{r['eval_risk']:.4f} | {'yes' if r['guarantee_ok'] else 'NO'} | "
                  f"{r['est_latency_ms_per_query']:.1f} |")
    (reports / "tables" / "table4_conformal.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- Console summary ----------------
    viol = df[~df.guarantee_ok]
    viol_mean = mean_df[~mean_df.guarantee_ok_mean]
    print("=" * 70)
    print("CONFORMAL RISK-CONTROLLED SELECTIVE RERANKING — SUMMARY")
    print("=" * 70)
    print(f"Eval split: Hybrid nDCG@10={hybrid_eval:.4f}  Always-Rerank nDCG@10={always_eval:.4f}  "
          f"(max gain {always_eval - hybrid_eval:+.4f})")
    print(f"Single-split (seed={args.seed}) risk>alpha points: {len(viol)} / {len(df)} "
          f"(expected: CRC controls risk only in expectation)")
    print(f"Mean-over-200-splits E[risk]>alpha points: {len(viol_mean)} / {len(mean_df)}  "
          f"(should be ~0 -> guarantee holds in expectation)")
    print(f"\nHeadline at alpha={a}:")
    for name, r in headline.items():
        print(f"  {name:11s}: coverage={r['eval_coverage']:.0%}  nDCG@10={r['eval_ndcg']:.4f}  "
              f"risk={r['eval_risk']:.4f}  ~{r['est_latency_ms_per_query']:.1f} ms/q")
    if "hybrid_max" in headline and "score_gap" in headline:
        dcov = headline["score_gap"]["eval_coverage"] - headline["hybrid_max"]["eval_coverage"]
        print(f"  -> QPP signal saves {dcov:+.0%} rerank coverage vs score-gap at the same guarantee")
    print(f"\nWrote: {run_dir / (args.split + '_conformal_results.csv')}")
    print(f"       {reports / 'figures' / 'phase3_conformal_risk_coverage.png'}")
    print(f"       {reports / 'tables' / 'table4_conformal.md'}")


if __name__ == "__main__":
    main()
