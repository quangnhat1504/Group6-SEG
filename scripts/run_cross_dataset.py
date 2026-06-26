"""Cross-dataset validation: Full SEG pipeline on NFCorpus.

Demonstrates generalizability of the SEG approach (BM25 + Dense fusion,
QPP-based selective reranking, conformal risk control) on a second BEIR
biomedical dataset.

NFCorpus: 3,633 documents, 323 test queries, multi-graded relevance.

Outputs:
  - data/nfcorpus/ (BEIR-format dataset)
  - runs/nfcorpus/test_*.csv
  - runs/nfcorpus/test_qpp_features.csv
  - runs/nfcorpus/test_conformal_results.csv
  - reports/tables/table_cross_dataset.md
  - reports/figures/cross_dataset_comparison.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats

from seg_retrieval.config import load_config
from seg_retrieval.datasets import load_beir_split_direct
from seg_retrieval.fusion import reciprocal_rank_fusion, top_doc_overlap
from seg_retrieval.io import load_run, save_run
from seg_retrieval.metrics import evaluate_run, per_query_ndcg
from seg_retrieval.qpp import corpus_mean, qpp_features
from seg_retrieval.rerank import CrossEncoderReranker
from seg_retrieval.retrievers import BM25Retriever, DenseRetriever
from seg_retrieval.types import Query

# Import CRC functions from run_conformal_rerank.py
sys.path.insert(0, str(Path(__file__).parent))
from run_conformal_rerank import crc_threshold, evaluate


# ---------------------------------------------------------------------------
# NFCorpus download / preparation
# ---------------------------------------------------------------------------

BEIR_ZIP_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"


def _prepare_nfcorpus_huggingface(data_dir: Path) -> bool:
    """Try downloading NFCorpus via HuggingFace datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [HF] datasets library not installed, skipping HuggingFace method.")
        return False

    print("  [HF] Downloading NFCorpus corpus via HuggingFace datasets...")
    try:
        ds_corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    except Exception as e:
        print(f"  [HF] Failed to load corpus: {e}")
        return False

    print("  [HF] Downloading NFCorpus queries...")
    try:
        ds_queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    except Exception as e:
        print(f"  [HF] Failed to load queries: {e}")
        return False

    print("  [HF] Downloading NFCorpus qrels...")
    try:
        ds_qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")
    except Exception as e:
        print(f"  [HF] Failed to load qrels: {e}")
        return False

    # Write corpus.jsonl
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = data_dir / "corpus.jsonl"
    print(f"  [HF] Writing {len(ds_corpus)} documents to {corpus_path}")
    with corpus_path.open("w", encoding="utf-8") as f:
        for row in ds_corpus:
            doc = {
                "_id": str(row["_id"]),
                "title": str(row.get("title", "") or ""),
                "text": str(row.get("text", "") or ""),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Write queries.jsonl
    queries_path = data_dir / "queries.jsonl"
    print(f"  [HF] Writing {len(ds_queries)} queries to {queries_path}")
    with queries_path.open("w", encoding="utf-8") as f:
        for row in ds_queries:
            q = {"_id": str(row["_id"]), "text": str(row.get("text", "") or "")}
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Write qrels/test.tsv
    qrels_dir = data_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    qrels_path = qrels_dir / "test.tsv"
    print(f"  [HF] Writing {len(ds_qrels)} qrel entries to {qrels_path}")
    with qrels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "score"])
        for row in ds_qrels:
            writer.writerow([str(row["query-id"]), str(row["corpus-id"]), int(row["score"])])

    print("  [HF] NFCorpus preparation complete.")
    return True


def _prepare_nfcorpus_beir_zip(data_dir: Path) -> bool:
    """Fallback: download BEIR zip and extract."""
    zip_path = data_dir.parent / "nfcorpus.zip"
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"  [ZIP] Downloading NFCorpus from {BEIR_ZIP_URL}...")
    try:
        urllib.request.urlretrieve(BEIR_ZIP_URL, zip_path)
    except Exception as e:
        print(f"  [ZIP] Download failed: {e}")
        return False

    print(f"  [ZIP] Extracting to {data_dir.parent}...")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_dir.parent)
    except Exception as e:
        print(f"  [ZIP] Extraction failed: {e}")
        return False

    print("  [ZIP] NFCorpus preparation complete.")
    return True


def prepare_nfcorpus(data_dir: Path) -> None:
    """Download and prepare NFCorpus data in BEIR format."""
    # Check if already prepared
    if (data_dir / "corpus.jsonl").exists() and (data_dir / "qrels" / "test.tsv").exists():
        print("NFCorpus data already exists, skipping download.")
        return

    print("Preparing NFCorpus dataset...")
    # Try HuggingFace first
    if _prepare_nfcorpus_huggingface(data_dir):
        return
    # Fallback to BEIR zip
    if _prepare_nfcorpus_beir_zip(data_dir):
        return
    # Both failed
    raise RuntimeError(
        "Failed to download NFCorpus via both methods.\n"
        f"  HuggingFace: pip install datasets\n"
        f"  BEIR zip: {BEIR_ZIP_URL}\n"
        "Please download manually and extract to data/nfcorpus/"
    )


# ---------------------------------------------------------------------------
# SciFact results loading (for comparison table)
# ---------------------------------------------------------------------------

def load_scifact_metrics() -> dict[str, dict[str, float]]:
    """Load existing SciFact run metrics for comparison."""
    scifact_run_dir = Path("runs/scifact")
    scifact_data_dir = Path("data/scifact")

    # Load SciFact qrels
    from seg_retrieval.io import load_qrels
    qrels = load_qrels(scifact_data_dir / "test_qrels.csv")

    metrics = {}
    run_names = ["test_bm25", "test_dense", "test_hybrid", "test_always_rerank"]
    for name in run_names:
        run_path = scifact_run_dir / f"{name}.csv"
        if run_path.exists():
            run = load_run(run_path)
            metrics[name.replace("test_", "")] = evaluate_run(run, qrels)

    return metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset validation: SEG pipeline on NFCorpus"
    )
    parser.add_argument("--config", default="configs/nfcorpus.yaml")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--alpha", type=float, default=0.02)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = config.dataset.data_dir
    run_dir = Path(config.outputs.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = Path("reports")
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Download/prepare NFCorpus ----
    print("=" * 70)
    print("CROSS-DATASET VALIDATION: NFCorpus")
    print("=" * 70)
    prepare_nfcorpus(data_dir)

    # ---- Step 2: Load corpus, queries, qrels ----
    print("\nLoading NFCorpus data...")
    documents, queries, qrels = load_beir_split_direct(data_dir, "test")
    print(f"  Corpus: {len(documents)} documents")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Qrels: {len(qrels)} judged queries")

    # Build document lookup for reranker
    doc_lookup = {doc.doc_id: doc for doc in documents}

    # ---- Step 3: BM25 retrieval ----
    print("\n[1/7] Running BM25 retrieval (top-100)...")
    t0 = time.perf_counter()
    bm25_retriever = BM25Retriever(documents)
    bm25_run = bm25_retriever.search(queries, config.retrieval.top_k)
    bm25_time = time.perf_counter() - t0
    save_run(run_dir / "test_bm25.csv", bm25_run)
    print(f"  BM25 done in {bm25_time:.1f}s, {len(bm25_run)} queries retrieved.")

    # ---- Step 4: Dense/SciNCL retrieval ----
    print("\n[2/7] Running Dense/SciNCL retrieval (top-100)...")
    t0 = time.perf_counter()
    dense_retriever = DenseRetriever(
        documents, config.retrieval.dense_model, config.retrieval.dense_batch_size
    )
    dense_run = dense_retriever.search(queries, config.retrieval.top_k)
    dense_time = time.perf_counter() - t0
    save_run(run_dir / "test_dense.csv", dense_run)
    print(f"  Dense done in {dense_time:.1f}s, {len(dense_run)} queries retrieved.")

    # ---- Step 5: Hybrid RRF fusion ----
    print("\n[3/7] Fusing with RRF (k=60, top_k=100)...")
    hybrid_run = reciprocal_rank_fusion(
        [bm25_run, dense_run], k=config.retrieval.rrf_k, top_k=config.retrieval.top_k
    )
    save_run(run_dir / "test_hybrid.csv", hybrid_run)
    print(f"  Hybrid fusion complete: {len(hybrid_run)} queries.")

    # ---- Step 6: Always-Rerank (top-20 for all queries) ----
    print("\n[4/7] Reranking top-20 for all {0} queries...".format(len(queries)))
    reranker = CrossEncoderReranker(config.rerank.model)
    rerank_run = {}
    t0 = time.perf_counter()
    for i, query in enumerate(queries, 1):
        hits = hybrid_run.get(query.query_id, [])
        reranked_hits = reranker.rerank(query, doc_lookup, hits, config.rerank.top_k)
        rerank_run[query.query_id] = reranked_hits
        if i % 50 == 0 or i == len(queries):
            print(f"    Reranked {i}/{len(queries)} queries...")
    rerank_time = time.perf_counter() - t0
    save_run(run_dir / "test_always_rerank.csv", rerank_run)
    print(f"  Reranking done in {rerank_time:.1f}s ({rerank_time/len(queries)*1000:.1f} ms/query).")

    # ---- Step 7: Evaluate all runs ----
    print("\n[5/7] Evaluating all runs...")
    metrics = {
        "bm25": evaluate_run(bm25_run, qrels),
        "dense": evaluate_run(dense_run, qrels),
        "hybrid": evaluate_run(hybrid_run, qrels),
        "always_rerank": evaluate_run(rerank_run, qrels),
    }
    for name, m in metrics.items():
        print(f"  {name:15s}: nDCG@10={m['ndcg@10']:.4f}  R@10={m['recall@10']:.4f}  "
              f"R@100={m['recall@100']:.4f}  MRR@10={m['mrr@10']:.4f}")

    # ---- Step 8: QPP features + correlations ----
    print("\n[6/7] Computing QPP features and correlations...")
    base_ndcg = per_query_ndcg(hybrid_run, qrels, 10)
    rerank_ndcg_pq = per_query_ndcg(rerank_run, qrels, 10)

    mu = {
        "bm25": corpus_mean(bm25_run),
        "dense": corpus_mean(dense_run),
        "hybrid": corpus_mean(hybrid_run),
    }

    qpp_rows = []
    for q in queries:
        qid = q.query_id
        feats: dict[str, object] = {"query_id": qid}
        feats.update(qpp_features(bm25_run.get(qid, []), mu["bm25"], "bm25", 10))
        feats.update(qpp_features(dense_run.get(qid, []), mu["dense"], "dense", 10))
        feats.update(qpp_features(hybrid_run.get(qid, []), mu["hybrid"], "hybrid", 10))
        overlap = top_doc_overlap(bm25_run.get(qid, []), dense_run.get(qid, []), k=10)
        feats["bm25_dense_overlap"] = overlap
        feats["disagreement"] = 1.0 - overlap
        feats["base_ndcg"] = base_ndcg.get(qid, 0.0)
        feats["rerank_ndcg"] = rerank_ndcg_pq.get(qid, 0.0)
        feats["gain"] = rerank_ndcg_pq.get(qid, 0.0) - base_ndcg.get(qid, 0.0)
        qpp_rows.append(feats)

    qpp_df = pd.DataFrame(qpp_rows)
    qpp_df.to_csv(run_dir / "test_qpp_features.csv", index=False)

    # Correlations
    feature_cols = [c for c in qpp_df.columns
                    if c not in ("query_id", "base_ndcg", "rerank_ndcg", "gain")]
    corr_rows = []
    for col in feature_cols:
        x = qpp_df[col].to_numpy()
        if np.std(x) < 1e-12:
            continue
        kt_gain = stats.kendalltau(x, qpp_df["gain"]).statistic
        pr_gain = stats.pearsonr(x, qpp_df["gain"]).statistic
        corr_rows.append({
            "feature": col,
            "kendall_vs_gain": kt_gain,
            "pearson_vs_gain": pr_gain,
            "abs_kendall_vs_gain": abs(kt_gain),
        })
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_kendall_vs_gain", ascending=False)
    best_predictor = corr_df.iloc[0]["feature"] if len(corr_df) > 0 else "N/A"
    best_kt = corr_df.iloc[0]["kendall_vs_gain"] if len(corr_df) > 0 else 0.0
    print(f"  Best QPP predictor for gain: {best_predictor} (Kendall={best_kt:+.3f})")

    # ---- Step 9: CRC with calibration/eval split ----
    print("\n[7/7] Running Conformal Risk Control (seed={0}, alpha={1})...".format(
        args.seed, args.alpha))
    ids = qpp_df["query_id"].tolist()
    base_ndcg_dict = dict(zip(qpp_df["query_id"], qpp_df["base_ndcg"]))
    rerank_ndcg_dict = dict(zip(qpp_df["query_id"], qpp_df["rerank_ndcg"]))
    loss = {q: max(0.0, rerank_ndcg_dict[q] - base_ndcg_dict[q]) for q in ids}

    # Use hybrid_max as signal (primary QPP trigger)
    signal = dict(zip(qpp_df["query_id"], qpp_df["hybrid_max"]))

    # Calibration / evaluation split
    rng = np.random.default_rng(args.seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_cal = len(ids) // 2
    cal_ids, eval_ids = shuffled[:n_cal], shuffled[n_cal:]

    lam = crc_threshold(signal, loss, cal_ids, args.alpha)
    ev = evaluate(signal, loss, base_ndcg_dict, rerank_ndcg_dict, eval_ids, lam)

    guarantee_ok = ev["risk"] <= args.alpha
    print(f"  Lambda: {lam:.6f}")
    print(f"  Eval coverage: {ev['coverage']:.2%}")
    print(f"  Eval nDCG@10: {ev['ndcg']:.4f}")
    print(f"  Eval risk: {ev['risk']:.4f}")
    print(f"  Guarantee (risk <= {args.alpha}): {'YES' if guarantee_ok else 'NO'}")

    # Save conformal results
    conformal_df = pd.DataFrame([{
        "signal": "hybrid_max",
        "alpha": args.alpha,
        "lambda": lam,
        "eval_coverage": ev["coverage"],
        "eval_ndcg": ev["ndcg"],
        "eval_risk": ev["risk"],
        "guarantee_ok": guarantee_ok,
        "n_cal": len(cal_ids),
        "n_eval": len(eval_ids),
    }])
    conformal_df.to_csv(run_dir / "test_conformal_results.csv", index=False)

    # ---- Step 10: Comparison table (SciFact vs NFCorpus) ----
    print("\nGenerating comparison table...")
    scifact_metrics = load_scifact_metrics()

    md_lines = [
        "# Cross-Dataset Comparison: SciFact vs NFCorpus",
        "",
        "| Metric | SciFact (300q) | NFCorpus (323q) |",
        "|--------|---------------:|----------------:|",
    ]

    metric_keys = ["ndcg@10", "recall@10", "recall@100", "mrr@10"]
    run_labels = [
        ("bm25", "BM25"),
        ("dense", "Dense/SciNCL"),
        ("hybrid", "Hybrid RRF k=60"),
        ("always_rerank", "Always-Rerank (top-20)"),
    ]

    for run_key, label in run_labels:
        for mk in metric_keys:
            sf_val = scifact_metrics.get(run_key, {}).get(mk, float("nan"))
            nf_val = metrics.get(run_key, {}).get(mk, float("nan"))
            md_lines.append(
                f"| {label} {mk} | {sf_val:.4f} | {nf_val:.4f} |"
            )

    md_lines.extend([
        "",
        "## QPP Best Predictor",
        f"- NFCorpus: **{best_predictor}** (Kendall vs gain = {best_kt:+.3f})",
        "",
        "## Conformal Risk Control (alpha={0}, seed={1})".format(args.alpha, args.seed),
        "",
        "| Dataset | Coverage | Selective nDCG@10 | Realized Risk | Guarantee |",
        "|---------|:--------:|:-----------------:|:-------------:|:---------:|",
    ])

    # NFCorpus CRC row
    md_lines.append(
        f"| NFCorpus | {ev['coverage']:.2%} | {ev['ndcg']:.4f} | "
        f"{ev['risk']:.4f} | {'Yes' if guarantee_ok else 'NO (violated)'} |"
    )

    # SciFact CRC row (load from existing results if available)
    sf_conformal_path = Path("runs/scifact/test_conformal_results.csv")
    if sf_conformal_path.exists():
        sf_crc = pd.read_csv(sf_conformal_path)
        sf_crc_row = sf_crc[
            (sf_crc["signal"] == "hybrid_max") &
            (np.isclose(sf_crc["alpha"], args.alpha))
        ]
        if len(sf_crc_row) > 0:
            r = sf_crc_row.iloc[0]
            md_lines.append(
                f"| SciFact | {r['eval_coverage']:.2%} | {r['eval_ndcg']:.4f} | "
                f"{r['eval_risk']:.4f} | {'Yes' if r['guarantee_ok'] else 'NO (violated)'} |"
            )

    # Report guarantee violation as finding
    if not guarantee_ok:
        md_lines.extend([
            "",
            "## Finding: CRC Guarantee Violation on NFCorpus",
            f"- The conformal guarantee (E[risk] <= alpha = {args.alpha}) was **violated**.",
            f"- Actual realized risk: {ev['risk']:.4f}",
            "- This may indicate the calibration set was too small or the signal "
            "does not generalize perfectly to NFCorpus.",
        ])

    md_lines.append("")
    table_path = reports / "tables" / "table_cross_dataset.md"
    table_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Wrote: {table_path}")

    # ---- Step 11: Cross-dataset comparison figure ----
    print("\nGenerating cross-dataset comparison figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: nDCG@10 comparison (grouped bar chart)
    ax = axes[0]
    labels_bar = ["BM25", "Dense", "Hybrid", "Always-Rerank"]
    run_keys_bar = ["bm25", "dense", "hybrid", "always_rerank"]
    sf_ndcg = [scifact_metrics.get(k, {}).get("ndcg@10", 0) for k in run_keys_bar]
    nf_ndcg = [metrics.get(k, {}).get("ndcg@10", 0) for k in run_keys_bar]

    x = np.arange(len(labels_bar))
    width = 0.35
    bars1 = ax.bar(x - width / 2, sf_ndcg, width, label="SciFact", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, nf_ndcg, width, label="NFCorpus", color="#ff7f0e")
    ax.set_xlabel("Run")
    ax.set_ylabel("nDCG@10")
    ax.set_title("nDCG@10: SciFact vs NFCorpus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Right panel: Recall@100 comparison
    ax = axes[1]
    sf_recall = [scifact_metrics.get(k, {}).get("recall@100", 0) for k in run_keys_bar]
    nf_recall = [metrics.get(k, {}).get("recall@100", 0) for k in run_keys_bar]

    bars3 = ax.bar(x - width / 2, sf_recall, width, label="SciFact", color="#1f77b4")
    bars4 = ax.bar(x + width / 2, nf_recall, width, label="NFCorpus", color="#ff7f0e")
    ax.set_xlabel("Run")
    ax.set_ylabel("Recall@100")
    ax.set_title("Recall@100: SciFact vs NFCorpus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    fig.suptitle("Cross-Dataset Generalizability: SciFact vs NFCorpus", fontsize=13)
    fig.tight_layout()
    fig_path = reports / "figures" / "cross_dataset_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote: {fig_path}")

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("CROSS-DATASET VALIDATION — SUMMARY")
    print("=" * 70)
    print(f"Dataset: NFCorpus ({len(documents)} docs, {len(queries)} queries)")
    print(f"BM25 nDCG@10:         {metrics['bm25']['ndcg@10']:.4f}")
    print(f"Dense nDCG@10:        {metrics['dense']['ndcg@10']:.4f}")
    print(f"Hybrid nDCG@10:       {metrics['hybrid']['ndcg@10']:.4f}")
    print(f"Always-Rerank nDCG@10:{metrics['always_rerank']['ndcg@10']:.4f}")
    print(f"Best QPP predictor:   {best_predictor} (Kendall={best_kt:+.3f})")
    print(f"CRC (alpha={args.alpha}): coverage={ev['coverage']:.2%}, "
          f"nDCG={ev['ndcg']:.4f}, risk={ev['risk']:.4f}, "
          f"guarantee={'OK' if guarantee_ok else 'VIOLATED'}")
    print(f"\nOutputs:")
    print(f"  {run_dir / 'test_bm25.csv'}")
    print(f"  {run_dir / 'test_dense.csv'}")
    print(f"  {run_dir / 'test_hybrid.csv'}")
    print(f"  {run_dir / 'test_always_rerank.csv'}")
    print(f"  {run_dir / 'test_qpp_features.csv'}")
    print(f"  {run_dir / 'test_conformal_results.csv'}")
    print(f"  {table_path}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
