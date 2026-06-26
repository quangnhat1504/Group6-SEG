"""Experiment 4: Full Phase-3 pipeline on the k=5 RRF base (stronger base ablation).

Re-fuses BM25 + Dense retrieval with RRF k=5 (instead of k=60), then runs the
full Always-Rerank + QPP + Conformal pipeline. A lower k value strengthens the
hybrid base by giving higher weight to top-ranked documents, testing whether
selective reranking remains beneficial even with a stronger starting point.

Outputs:
  - runs/scifact/test_hybrid_k5.csv
  - runs/scifact/test_always_rerank_k5.csv
  - runs/scifact/test_qpp_features_k5.csv
  - runs/scifact/test_conformal_results_k5.csv
  - reports/tables/table_k5_comparison.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats

from seg_retrieval.config import load_config
from seg_retrieval.fusion import reciprocal_rank_fusion, top_doc_overlap
from seg_retrieval.io import load_documents, load_qrels, load_queries, load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.qpp import corpus_mean, qpp_features
from seg_retrieval.rerank import CrossEncoderReranker

# Import CRC functions from run_conformal_rerank.py
sys.path.insert(0, str(Path(__file__).parent))
from run_conformal_rerank import crc_threshold, evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Phase-3 pipeline on RRF k=5 base (stronger base ablation)."
    )
    parser.add_argument("--config", default="configs/scifact.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rrf-k", type=int, default=5, help="RRF k parameter (default: 5).")
    parser.add_argument("--seed", type=int, default=13, help="Calibration/eval split seed.")
    parser.add_argument("--alpha", type=float, default=0.02, help="CRC risk budget.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = config.outputs.run_dir
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"RRF k={args.rrf_k} FULL PIPELINE — SciFact {args.split}")
    print("=" * 70)

    # ----------------------------------------------------------------
    # Step 1: Re-fuse BM25 + Dense with k=5
    # ----------------------------------------------------------------
    print(f"\n[1/7] Loading BM25 and Dense runs, fusing with k={args.rrf_k}...")
    bm25 = load_run(run_dir / f"{args.split}_bm25.csv")
    dense = load_run(run_dir / f"{args.split}_dense.csv")
    hybrid_k5 = reciprocal_rank_fusion([bm25, dense], k=args.rrf_k, top_k=100)
    save_run(run_dir / f"{args.split}_hybrid_k{args.rrf_k}.csv", hybrid_k5)
    print(f"  Fused {len(hybrid_k5)} queries → {run_dir / f'{args.split}_hybrid_k{args.rrf_k}.csv'}")

    # ----------------------------------------------------------------
    # Step 2: Rerank top-20 of k=5 hybrid for all queries
    # ----------------------------------------------------------------
    print("\n[2/7] Loading documents and reranking top-20 for all queries...")
    documents = load_documents(config.dataset.data_dir / f"{args.split}_documents.jsonl")
    doc_map = {doc.doc_id: doc for doc in documents}
    queries = load_queries(config.dataset.data_dir / f"{args.split}_queries.jsonl")
    qrels = load_qrels(config.dataset.data_dir / f"{args.split}_qrels.csv")

    reranker = CrossEncoderReranker(model_name=config.rerank.model)
    reranked_run = {}
    t0 = time.perf_counter()
    for q in queries:
        hits = hybrid_k5.get(q.query_id, [])
        reranked_run[q.query_id] = reranker.rerank(q, doc_map, hits, top_k=20)
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / len(queries)) * 1000 if queries else 0.0

    save_run(run_dir / f"{args.split}_always_rerank_k{args.rrf_k}.csv", reranked_run)
    print(f"  Reranked {len(queries)} queries in {elapsed:.1f}s ({latency_ms:.1f} ms/query)")
    print(f"  → {run_dir / f'{args.split}_always_rerank_k{args.rrf_k}.csv'}")

    # ----------------------------------------------------------------
    # Step 3: Evaluate k=5 hybrid and k=5 always-rerank
    # ----------------------------------------------------------------
    print("\n[3/7] Evaluating k=5 runs...")
    metrics_hybrid_k5 = evaluate_run(hybrid_k5, qrels)
    metrics_rerank_k5 = evaluate_run(reranked_run, qrels)
    print(f"  Hybrid k={args.rrf_k}:       nDCG@10={metrics_hybrid_k5['ndcg@10']:.4f}  "
          f"Recall@100={metrics_hybrid_k5['recall@100']:.4f}")
    print(f"  Always-Rerank k={args.rrf_k}: nDCG@10={metrics_rerank_k5['ndcg@10']:.4f}  "
          f"Recall@100={metrics_rerank_k5['recall@100']:.4f}")

    # ----------------------------------------------------------------
    # Step 4: Compute QPP features on k=5 base
    # ----------------------------------------------------------------
    print("\n[4/7] Computing QPP features on k=5 base...")
    base_ndcg = per_query_ndcg(hybrid_k5, qrels, 10)
    rerank_ndcg = per_query_ndcg(reranked_run, qrels, 10)

    mu = {"bm25": corpus_mean(bm25), "dense": corpus_mean(dense), "hybrid": corpus_mean(hybrid_k5)}

    rows = []
    for q in queries:
        qid = q.query_id
        feats: dict[str, float] = {"query_id": qid}
        feats.update(qpp_features(bm25.get(qid, []), mu["bm25"], "bm25", 10))
        feats.update(qpp_features(dense.get(qid, []), mu["dense"], "dense", 10))
        feats.update(qpp_features(hybrid_k5.get(qid, []), mu["hybrid"], "hybrid", 10))
        overlap = top_doc_overlap(bm25.get(qid, []), dense.get(qid, []), k=10)
        feats["bm25_dense_overlap"] = overlap
        feats["disagreement"] = 1.0 - overlap
        feats["base_ndcg"] = base_ndcg.get(qid, 0.0)
        feats["rerank_ndcg"] = rerank_ndcg.get(qid, 0.0)
        feats["gain"] = rerank_ndcg.get(qid, 0.0) - base_ndcg.get(qid, 0.0)
        rows.append(feats)

    feats_df = pd.DataFrame(rows)
    feats_df.to_csv(run_dir / f"{args.split}_qpp_features_k{args.rrf_k}.csv", index=False)
    print(f"  → {run_dir / f'{args.split}_qpp_features_k{args.rrf_k}.csv'}")

    # Check if hybrid_max remains top signal
    feature_cols = [c for c in feats_df.columns
                    if c not in ("query_id", "base_ndcg", "rerank_ndcg", "gain")]
    corr_rows = []
    for col in feature_cols:
        x = feats_df[col].to_numpy()
        if np.std(x) < 1e-12:
            continue
        kt_gain = stats.kendalltau(x, feats_df["gain"]).statistic
        corr_rows.append({"feature": col, "kendall_vs_gain": kt_gain,
                          "abs_kendall_vs_gain": abs(kt_gain)})
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_kendall_vs_gain", ascending=False)
    top_signal = corr_df.iloc[0]["feature"] if len(corr_df) > 0 else "hybrid_max"
    print(f"  Top QPP signal for gain: {top_signal} "
          f"(|Kendall|={corr_df.iloc[0]['abs_kendall_vs_gain']:.3f})")
    if top_signal != "hybrid_max":
        hm_row = corr_df[corr_df.feature == "hybrid_max"]
        if len(hm_row):
            print(f"  hybrid_max |Kendall|={hm_row.iloc[0]['abs_kendall_vs_gain']:.3f} "
                  f"(not top on k={args.rrf_k})")

    # ----------------------------------------------------------------
    # Step 5: Run CRC (150/150 split, seed=13, alpha=0.02)
    # ----------------------------------------------------------------
    print(f"\n[5/7] Running CRC (seed={args.seed}, alpha={args.alpha})...")
    ids = feats_df["query_id"].tolist()
    base_ndcg_dict = dict(zip(feats_df["query_id"], feats_df["base_ndcg"]))
    rerank_ndcg_dict = dict(zip(feats_df["query_id"], feats_df["rerank_ndcg"]))
    loss = {q: max(0.0, rerank_ndcg_dict[q] - base_ndcg_dict[q]) for q in ids}

    # Use hybrid_max as the signal (consistent with main pipeline)
    signal = dict(zip(feats_df["query_id"], feats_df["hybrid_max"]))

    rng = np.random.default_rng(args.seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_cal = len(ids) // 2  # 150/150 split
    cal_ids, eval_ids = shuffled[:n_cal], shuffled[n_cal:]

    # Sweep alpha range for completeness
    alphas = np.round(np.arange(0.005, 0.0501, 0.0025), 4)
    crc_rows = []
    for alpha_val in alphas:
        lam = crc_threshold(signal, loss, cal_ids, float(alpha_val))
        ev = evaluate(signal, loss, base_ndcg_dict, rerank_ndcg_dict, eval_ids, lam)
        crc_rows.append({
            "signal": "hybrid_max", "alpha": float(alpha_val), "lambda": lam,
            "eval_coverage": ev["coverage"], "eval_ndcg": ev["ndcg"],
            "eval_risk": ev["risk"], "guarantee_ok": ev["risk"] <= alpha_val,
        })

    crc_df = pd.DataFrame(crc_rows)
    crc_df.to_csv(run_dir / f"{args.split}_conformal_results_k{args.rrf_k}.csv", index=False)
    print(f"  → {run_dir / f'{args.split}_conformal_results_k{args.rrf_k}.csv'}")

    # Get headline result at target alpha
    headline = crc_df[np.isclose(crc_df["alpha"], args.alpha)]
    if len(headline):
        h = headline.iloc[0]
        print(f"  At alpha={args.alpha}: coverage={h['eval_coverage']:.2f}, "
              f"nDCG@10={h['eval_ndcg']:.4f}, risk={h['eval_risk']:.4f}, "
              f"guarantee={'OK' if h['guarantee_ok'] else 'VIOLATED'}")
        selective_ndcg_k5 = h["eval_ndcg"]
        selective_coverage_k5 = h["eval_coverage"]
    else:
        selective_ndcg_k5 = None
        selective_coverage_k5 = None

    # ----------------------------------------------------------------
    # Step 6: Load existing k=60 results for comparison
    # ----------------------------------------------------------------
    print("\n[6/7] Loading k=60 results for comparison...")
    hybrid_k60 = load_run(run_dir / f"{args.split}_hybrid.csv")
    reranked_k60 = load_run(run_dir / f"{args.split}_always_rerank.csv")
    metrics_hybrid_k60 = evaluate_run(hybrid_k60, qrels)
    metrics_rerank_k60 = evaluate_run(reranked_k60, qrels)

    # Load k=60 conformal results
    conformal_k60_path = run_dir / f"{args.split}_conformal_results.csv"
    if conformal_k60_path.exists():
        conformal_k60 = pd.read_csv(conformal_k60_path)
        headline_k60 = conformal_k60[
            (conformal_k60["signal"] == "hybrid_max") &
            (np.isclose(conformal_k60["alpha"], args.alpha))
        ]
        if len(headline_k60):
            h60 = headline_k60.iloc[0]
            selective_ndcg_k60 = h60["eval_ndcg"]
            selective_coverage_k60 = h60["eval_coverage"]
        else:
            selective_ndcg_k60 = None
            selective_coverage_k60 = None
    else:
        selective_ndcg_k60 = None
        selective_coverage_k60 = None
        print("  WARNING: k=60 conformal results not found")

    print(f"  Hybrid k=60:       nDCG@10={metrics_hybrid_k60['ndcg@10']:.4f}")
    print(f"  Always-Rerank k=60: nDCG@10={metrics_rerank_k60['ndcg@10']:.4f}")
    if selective_ndcg_k60 is not None:
        print(f"  Selective k=60:     nDCG@10={selective_ndcg_k60:.4f} "
              f"(coverage={selective_coverage_k60:.2f})")

    # ----------------------------------------------------------------
    # Step 7: Produce comparison table and note findings
    # ----------------------------------------------------------------
    print("\n[7/7] Producing comparison table...")

    # Compute always-rerank nDCG on eval subset for fair comparison
    always_rerank_eval_k5 = sum(rerank_ndcg_dict[q] for q in eval_ids) / len(eval_ids)
    hybrid_eval_k5 = sum(base_ndcg_dict[q] for q in eval_ids) / len(eval_ids)

    # k=60 eval subset comparison
    feats_k60_path = run_dir / f"{args.split}_qpp_features.csv"
    if feats_k60_path.exists():
        feats_k60_df = pd.read_csv(feats_k60_path)
        feats_k60_df["query_id"] = feats_k60_df["query_id"].astype(str)
        base_ndcg_k60 = dict(zip(feats_k60_df["query_id"], feats_k60_df["base_ndcg"]))
        rerank_ndcg_k60 = dict(zip(feats_k60_df["query_id"], feats_k60_df["rerank_ndcg"]))
        always_rerank_eval_k60 = sum(rerank_ndcg_k60.get(q, 0.0) for q in eval_ids) / len(eval_ids)
        hybrid_eval_k60 = sum(base_ndcg_k60.get(q, 0.0) for q in eval_ids) / len(eval_ids)
    else:
        always_rerank_eval_k60 = metrics_rerank_k60["ndcg@10"]
        hybrid_eval_k60 = metrics_hybrid_k60["ndcg@10"]

    # Build markdown comparison table
    md_lines = [
        f"# RRF k=5 vs k=60 Pipeline Comparison",
        "",
        f"Evaluation on SciFact {args.split} split ({len(queries)} queries total, "
        f"{len(eval_ids)} eval subset). CRC: seed={args.seed}, alpha={args.alpha}, "
        f"signal=hybrid_max.",
        "",
        "## Full-Set Metrics (all queries)",
        "",
        "| Metric | k=60 | k=5 | Δ (k5 − k60) |",
        "|--------|-----:|----:|-------------:|",
        f"| Hybrid nDCG@10 | {metrics_hybrid_k60['ndcg@10']:.4f} | "
        f"{metrics_hybrid_k5['ndcg@10']:.4f} | "
        f"{metrics_hybrid_k5['ndcg@10'] - metrics_hybrid_k60['ndcg@10']:+.4f} |",
        f"| Hybrid Recall@100 | {metrics_hybrid_k60['recall@100']:.4f} | "
        f"{metrics_hybrid_k5['recall@100']:.4f} | "
        f"{metrics_hybrid_k5['recall@100'] - metrics_hybrid_k60['recall@100']:+.4f} |",
        f"| Always-Rerank nDCG@10 | {metrics_rerank_k60['ndcg@10']:.4f} | "
        f"{metrics_rerank_k5['ndcg@10']:.4f} | "
        f"{metrics_rerank_k5['ndcg@10'] - metrics_rerank_k60['ndcg@10']:+.4f} |",
        f"| Always-Rerank MRR@10 | {metrics_rerank_k60['mrr@10']:.4f} | "
        f"{metrics_rerank_k5['mrr@10']:.4f} | "
        f"{metrics_rerank_k5['mrr@10'] - metrics_rerank_k60['mrr@10']:+.4f} |",
        "",
        "## Eval-Subset Metrics (CRC evaluation split)",
        "",
        "| Metric | k=60 | k=5 | Δ (k5 − k60) |",
        "|--------|-----:|----:|-------------:|",
    ]

    # Add eval subset rows
    md_lines.append(
        f"| Hybrid nDCG@10 (eval) | {hybrid_eval_k60:.4f} | "
        f"{hybrid_eval_k5:.4f} | {hybrid_eval_k5 - hybrid_eval_k60:+.4f} |"
    )
    md_lines.append(
        f"| Always-Rerank nDCG@10 (eval) | {always_rerank_eval_k60:.4f} | "
        f"{always_rerank_eval_k5:.4f} | "
        f"{always_rerank_eval_k5 - always_rerank_eval_k60:+.4f} |"
    )
    if selective_ndcg_k5 is not None and selective_ndcg_k60 is not None:
        md_lines.append(
            f"| Conformal Selective nDCG@10 | {selective_ndcg_k60:.4f} | "
            f"{selective_ndcg_k5:.4f} | "
            f"{selective_ndcg_k5 - selective_ndcg_k60:+.4f} |"
        )
    if selective_coverage_k5 is not None and selective_coverage_k60 is not None:
        md_lines.append(
            f"| Rerank Coverage | {selective_coverage_k60:.2f} | "
            f"{selective_coverage_k5:.2f} | "
            f"{selective_coverage_k5 - selective_coverage_k60:+.2f} |"
        )

    # QPP signal comparison
    md_lines.extend([
        "",
        "## QPP Signal Analysis",
        "",
        f"- Top QPP signal for gain-from-reranking on k={args.rrf_k}: **{top_signal}** "
        f"(|Kendall|={corr_df.iloc[0]['abs_kendall_vs_gain']:.3f})",
    ])
    if top_signal == "hybrid_max":
        md_lines.append(
            f"- hybrid_max remains the top predictor on k={args.rrf_k} base (consistent with k=60)"
        )
    else:
        hm_row = corr_df[corr_df.feature == "hybrid_max"]
        hm_kt = hm_row.iloc[0]["abs_kendall_vs_gain"] if len(hm_row) else 0.0
        md_lines.append(
            f"- hybrid_max is NOT the top predictor on k={args.rrf_k} "
            f"(|Kendall|={hm_kt:.3f}); {top_signal} is better"
        )

    # Key finding: does selective on k=5 surpass always-rerank on k=5?
    md_lines.extend(["", "## Key Findings", ""])
    if selective_ndcg_k5 is not None:
        if selective_ndcg_k5 > always_rerank_eval_k5:
            md_lines.append(
                f"**CONFIRMATION**: Conformal selective reranking on k={args.rrf_k} "
                f"(nDCG@10={selective_ndcg_k5:.4f}) surpasses Always-Rerank on k={args.rrf_k} "
                f"(nDCG@10={always_rerank_eval_k5:.4f}) by "
                f"{selective_ndcg_k5 - always_rerank_eval_k5:+.4f}. "
                f"This confirms selective reranking can improve over always-reranking "
                f"regardless of base retriever strength."
            )
        else:
            md_lines.append(
                f"Conformal selective reranking on k={args.rrf_k} "
                f"(nDCG@10={selective_ndcg_k5:.4f}) does not surpass Always-Rerank on k={args.rrf_k} "
                f"(nDCG@10={always_rerank_eval_k5:.4f}). Δ = "
                f"{selective_ndcg_k5 - always_rerank_eval_k5:+.4f}."
            )

    if metrics_hybrid_k5["ndcg@10"] > metrics_hybrid_k60["ndcg@10"]:
        md_lines.append(
            f"\nk={args.rrf_k} produces a stronger hybrid base than k=60 "
            f"(+{metrics_hybrid_k5['ndcg@10'] - metrics_hybrid_k60['ndcg@10']:.4f} nDCG@10), "
            f"confirming the RRF k-dilution finding from Al-Joofi et al."
        )

    table_path = reports / "tables" / f"table_k{args.rrf_k}_comparison.md"
    table_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"  → {table_path}")

    # ----------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"RRF k={args.rrf_k} PIPELINE — SUMMARY")
    print("=" * 70)
    print(f"  Hybrid k={args.rrf_k} nDCG@10:       {metrics_hybrid_k5['ndcg@10']:.4f} "
          f"(k=60: {metrics_hybrid_k60['ndcg@10']:.4f})")
    print(f"  Always-Rerank k={args.rrf_k} nDCG@10: {metrics_rerank_k5['ndcg@10']:.4f} "
          f"(k=60: {metrics_rerank_k60['ndcg@10']:.4f})")
    if selective_ndcg_k5 is not None:
        print(f"  Selective k={args.rrf_k} nDCG@10:     {selective_ndcg_k5:.4f} "
              f"(coverage={selective_coverage_k5:.2f})")
    print(f"  Top QPP signal: {top_signal}")
    print(f"  Rerank latency: {latency_ms:.1f} ms/query")
    print(f"\nOutputs:")
    print(f"  {run_dir / f'{args.split}_hybrid_k{args.rrf_k}.csv'}")
    print(f"  {run_dir / f'{args.split}_always_rerank_k{args.rrf_k}.csv'}")
    print(f"  {run_dir / f'{args.split}_qpp_features_k{args.rrf_k}.csv'}")
    print(f"  {run_dir / f'{args.split}_conformal_results_k{args.rrf_k}.csv'}")
    print(f"  {table_path}")


if __name__ == "__main__":
    main()
