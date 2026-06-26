"""Direction 3 diagnostic: why downstream-utility reranking does not help on SciFact.

Offline analysis (no GPU) over the cached label distributions produced by
run_utility_rerank.py. Quantifies how well the small-LLM claim-verification signal
separates relevant from non-relevant abstracts, and shows that using it to rerank --
as a pure reorder, a blend with the base score, or a conformal-gated selective stage
-- never beats the Hybrid base. This is the evidence behind the negative result in
report Section 6.10.

Inputs (under runs/<dataset>/):
  - <split>_utility_cache.jsonl      per (query, doc) SUPPORT/REFUTE/NEI distribution
  - <split>_utility_baseline.jsonl   per query claim-only distribution (InfoGain prior)
  - <split>_hybrid.csv, <split>_qpp_features.csv
Outputs:
  - reports/tables/table6_utility_rerank.md
  - reports/figures/h3_utility_separation.png
  - reports/figures/h3_utility_selective_coverage.png
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from seg_retrieval.config import load_config
from seg_retrieval.io import load_qrels, load_run
from seg_retrieval.metrics import ndcg_at_k
from seg_retrieval.utility_rerank import (
    LABELS,
    information_gain,
    semantic_entropy,
    verification_confidence,
)

UNIFORM = {l: 1.0 / len(LABELS) for l in LABELS}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random positive scores above random negative) via rank-sum (ties = 0.5)."""
    if not pos or not neg:
        return float("nan")
    negs = sorted(neg)
    s = 0.0
    for v in pos:
        lo = bisect.bisect_left(negs, v)
        hi = bisect.bisect_right(negs, v)
        s += lo + (hi - lo) / 2.0
    return s / (len(pos) * len(neg))


def load_dist_jsonl(path: Path, key: str) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = str(r[key]) if key == "query_id" else (str(r["query_id"]), str(r["doc_id"]))
            out[k] = {l: float(r[l]) for l in LABELS}
    return out


def norm_base_scores(hits, k):
    n = min(len(hits), k)
    return {d: (n - i) / n for i, (d, _) in enumerate(hits[:k])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    k = args.top_k

    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")
    hybrid = load_run(run_dir / f"{args.split}_hybrid.csv")
    cache = load_dist_jsonl(run_dir / f"{args.split}_utility_cache.jsonl", "pair")
    base = load_dist_jsonl(run_dir / f"{args.split}_utility_baseline.jsonl", "query_id")
    feats = pd.read_csv(run_dir / f"{args.split}_qpp_features.csv")
    feats["query_id"] = feats["query_id"].astype(str)
    hmax = dict(zip(feats["query_id"], feats["hybrid_max"]))

    qids = [q for q in qrels if q in hybrid and q in base and q in hmax]

    # ---------- 1. Separation: signal value on relevant vs non-relevant candidates ----------
    signals = {
        "confidence": lambda qid, dd: verification_confidence(dd),
        "neg_entropy": lambda qid, dd: -semantic_entropy(dd),
        "infogain_decision": lambda qid, dd: information_gain(base[qid], dd, "decision"),
        "infogain_entropy": lambda qid, dd: information_gain(base[qid], dd, "entropy"),
    }
    sep = {}
    rel_conf, non_conf, rel_ent, non_ent = [], [], [], []
    for name, fn in signals.items():
        rel, non = [], []
        for (qid, did), dd in cache.items():
            if qid not in base:
                continue
            val = fn(qid, dd)
            is_rel = qrels.get(qid, {}).get(did, 0) > 0
            (rel if is_rel else non).append(val)
            if name == "confidence":
                (rel_conf if is_rel else non_conf).append(verification_confidence(dd))
                (rel_ent if is_rel else non_ent).append(semantic_entropy(dd))
        sep[name] = {"auc": auc(rel, non), "rel_mean": np.mean(rel), "non_mean": np.mean(non),
                     "n_rel": len(rel), "n_non": len(non)}

    # ---------- 2. Reranking variants ----------
    def base_ndcg(ids):
        return sum(ndcg_at_k({q: hybrid[q]}, {q: qrels[q]}, 10) for q in ids) / len(ids)

    def utility_run(score_fn, ids, w=1.0):
        run = {}
        for qid in ids:
            hits = hybrid[qid]
            nb = norm_base_scores(hits, k)
            sc = [(d, (1 - w) * nb[d] + w * score_fn(qid, cache.get((qid, d), UNIFORM)))
                  for d, _ in hits[:k] if d in nb]
            sc.sort(key=lambda x: -x[1])
            rest = [h for h in hits if h[0] not in {d for d, _ in hits[:k]}]
            run[qid] = sc + rest
        return run

    def run_ndcg(run, ids):
        return sum(ndcg_at_k({q: run[q]}, {q: qrels[q]}, 10) for q in ids) / len(ids)

    base_all = base_ndcg(qids)
    # pure-replace and best blend for the InfoGain-decision signal (the strongest)
    ig = signals["infogain_decision"]
    pure = run_ndcg(utility_run(ig, qids, w=1.0), qids)
    blend = [(w, run_ndcg(utility_run(ig, qids, w=w), qids))
             for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]]
    best_w, best_blend = max(blend, key=lambda x: x[1])
    conf_pure = run_ndcg(utility_run(signals["confidence"], qids, w=1.0), qids)

    # ---------- 3. Selective: gate by hybrid_max (low = uncertain), pure IG on triggered ----------
    order = sorted(qids, key=lambda q: hmax[q])  # most uncertain first
    bnd = {q: ndcg_at_k({q: hybrid[q]}, {q: qrels[q]}, 10) for q in qids}
    ind = {q: ndcg_at_k({q: utility_run(ig, [q], w=1.0)[q]}, {q: qrels[q]}, 10) for q in qids}
    sel_rows = []
    for cov in [0.1, 0.2, 0.3, 0.4, 0.5, 0.61, 0.8, 1.0]:
        n = max(1, int(round(cov * len(qids))))
        trig = set(order[:n])
        overall = sum((ind[q] if q in trig else bnd[q]) for q in qids) / len(qids)
        sel_rows.append({"coverage": cov, "n": n,
                         "selective_ndcg": overall,
                         "trig_base": sum(bnd[q] for q in trig) / n,
                         "trig_ig": sum(ind[q] for q in trig) / n})

    # ---------- Figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.hist(non_conf, bins=30, alpha=0.6, density=True, color="#d62728", label=f"non-relevant (n={len(non_conf)})")
    ax1.hist(rel_conf, bins=30, alpha=0.6, density=True, color="#2ca02c", label=f"relevant (n={len(rel_conf)})")
    ax1.set_xlabel("Verification confidence  max·(1−P(NEI))")
    ax1.set_ylabel("density")
    ax1.set_title(f"Confidence barely separates relevance\nAUC = {sep['confidence']['auc']:.3f}")
    ax1.legend(fontsize=8)
    covs = [r["coverage"] for r in sel_rows]
    ax2.axhline(base_all, ls="--", color="#888", label=f"Hybrid base = {base_all:.3f}")
    ax2.plot(covs, [r["selective_ndcg"] for r in sel_rows], marker="o", color="#1f77b4",
             label="selective utility-rerank")
    ax2.set_xlabel("Rerank coverage (fraction of most-uncertain queries)")
    ax2.set_ylabel("nDCG@10")
    ax2.set_title("Gating the utility reranker never beats base")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle("Direction 3: small-LLM downstream-utility reranking on SciFact (negative result)", fontsize=12)
    fig.tight_layout()
    fig.savefig(reports / "figures" / "h3_utility_separation.png", dpi=150)
    # second standalone figure: entropy separation
    fig2, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(non_ent, bins=30, alpha=0.6, density=True, color="#d62728", label="non-relevant")
    ax.hist(rel_ent, bins=30, alpha=0.6, density=True, color="#2ca02c", label="relevant")
    ax.set_xlabel("Semantic entropy of SUPPORT/REFUTE/NEI (normalized)")
    ax.set_ylabel("density")
    ax.set_title(f"Relevant abstracts are only slightly more decisive\nAUC(−entropy) = {sep['neg_entropy']['auc']:.3f}")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(reports / "figures" / "h3_utility_entropy.png", dpi=150)

    # ---------- Table ----------
    md = [
        "Direction 3: downstream-utility reranking with Qwen2.5-0.5B-Instruct on SciFact "
        f"({args.split}, {len(qids)} queries, top-{k} candidates). Negative result.",
        "",
        "Signal separation of relevant vs non-relevant candidates (higher AUC = more useful):",
        "",
        "| Signal | AUC | mean (relevant) | mean (non-relevant) |",
        "|---|---:|---:|---:|",
    ]
    for name in signals:
        s = sep[name]
        md.append(f"| {name} | {s['auc']:.3f} | {s['rel_mean']:.3f} | {s['non_mean']:.3f} |")
    md += [
        "",
        "Reranking nDCG@10 (base Hybrid RRF = "
        f"{base_all:.4f}); the LLM-utility signal never improves it:",
        "",
        "| Variant | nDCG@10 | Δ vs base |",
        "|---|---:|---:|",
        f"| Hybrid base (no rerank) | {base_all:.4f} | — |",
        f"| Always-Utility, confidence (pure reorder) | {conf_pure:.4f} | {conf_pure - base_all:+.4f} |",
        f"| Always-Utility, InfoGain-decision (pure reorder) | {pure:.4f} | {pure - base_all:+.4f} |",
        f"| Best blend, InfoGain-decision (w={best_w}) | {best_blend:.4f} | {best_blend - base_all:+.4f} |",
        f"| Selective, InfoGain-decision @61% coverage | "
        f"{[r['selective_ndcg'] for r in sel_rows if abs(r['coverage']-0.61)<1e-6][0]:.4f} | "
        f"{[r['selective_ndcg'] for r in sel_rows if abs(r['coverage']-0.61)<1e-6][0] - base_all:+.4f} |",
        "",
        "For comparison, the cross-encoder Always-Rerank reaches nDCG@10 ≈ 0.728 (Section 6.9): "
        "a purpose-built relevance reranker succeeds where the generative verifier does not.",
    ]
    (reports / "tables" / "table6_utility_rerank.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------- Console ----------
    print("=" * 70)
    print("DIRECTION 3 — DOWNSTREAM-UTILITY RERANKING (DIAGNOSTIC)")
    print("=" * 70)
    print(f"Queries: {len(qids)}  base nDCG@10: {base_all:.4f}")
    for name in signals:
        s = sep[name]
        print(f"  {name:18s} AUC={s['auc']:.3f}  rel={s['rel_mean']:.3f} non={s['non_mean']:.3f}")
    print(f"\nAlways-Utility (confidence) : {conf_pure:.4f} ({conf_pure - base_all:+.4f})")
    print(f"Always-Utility (InfoGain)   : {pure:.4f} ({pure - base_all:+.4f})")
    print(f"Best blend (InfoGain, w={best_w}) : {best_blend:.4f} ({best_blend - base_all:+.4f})")
    print("\nSelective (gate by hybrid_max, pure InfoGain on triggered):")
    for r in sel_rows:
        print(f"  cov={r['coverage']:.2f} n={r['n']:3d} | overall={r['selective_ndcg']:.4f} | "
              f"triggered base={r['trig_base']:.3f} -> IG={r['trig_ig']:.3f} ({r['trig_ig']-r['trig_base']:+.3f})")
    print(f"\nWrote: {reports / 'tables' / 'table6_utility_rerank.md'}")
    print(f"       {reports / 'figures' / 'h3_utility_separation.png'}")
    print(f"       {reports / 'figures' / 'h3_utility_entropy.png'}")


if __name__ == "__main__":
    main()
