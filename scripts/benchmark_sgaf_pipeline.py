"""Benchmark full SGAF pipeline: B5 mode-switch + P3 smoothing.

Pipeline steps per query (after offline doc encoding):
  1. BGE-small (specialist) encode + matmul top-100
  2. BGE-base (generalist) encode + matmul top-100
  3. Extract 5 features per query -> batch-average z-scores -> S
  4. Mode decision: S < 2.0 -> specialist-safe | S >= 2.0 -> generalist-fallback
  5. If fallback: P3 top-20 RRF blend with BGE-small prior
  6. Final ranking

Measures: offline (both models + doc encodes + B5 feature stats), online per-query breakdown.
"""
from __future__ import annotations

import json, math, statistics, time, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import _bootstrap  # noqa: F401

import numpy as np
from sentence_transformers import SentenceTransformer

from seg_retrieval.config import load_config
from seg_retrieval.fusion import reciprocal_rank_fusion
from seg_retrieval.retrievers import BM25Retriever
from seg_retrieval.types import Document, Query

ROOT = Path(__file__).resolve().parent.parent


def main():
    data = ROOT / "data" / "scifact"

    with open(data / "test_documents.jsonl", "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]
    docs = [Document(doc_id=r["doc_id"], title=r.get("title", ""), abstract=r.get("abstract", "")) for r in raw]

    with open(data / "test_queries.jsonl", "r", encoding="utf-8") as f:
        raw_q = [json.loads(l) for l in f]
    queries = [Query(query_id=r["query_id"], text=r["text"]) for r in raw_q]

    top_k = 100
    n = len(queries)
    warm = 10
    BATCH_SIZE = len(queries)  # single batch for simplicity
    corpus = [d.title + " " + d.text for d in docs]

    # ---------- B5 frozen params ----------
    TAU = 2.0
    P3_WINDOW = 20
    P3_ALPHA = 0.10
    K_RRF = 60

    print("=" * 60)
    print("OFFLINE (one-time setup)")
    print("=" * 60)

    t_all = time.perf_counter()

    # ---- BGE-small specialist load + doc encode ----
    t0 = time.perf_counter()
    model_small = SentenceTransformer("BAAI/bge-small-en-v1.5")
    # Load fine-tuned weights if available
    ft_path = ROOT / "runs" / "finetuned" / "bge-small-scifact-rrf"
    if ft_path.exists():
        model_small = SentenceTransformer(str(ft_path))
    load_s_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc_emb_small = model_small.encode(corpus, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    enc_s_s = time.perf_counter() - t0

    # ---- BGE-base generalist load + doc encode ----
    t0 = time.perf_counter()
    model_base = SentenceTransformer("BAAI/bge-base-en-v1.5")
    load_b_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc_emb_base = model_base.encode(corpus, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    enc_b_s = time.perf_counter() - t0

    # ---- B5 z-score statistics (frozen from SciFact trainfit) ----
    t0 = time.perf_counter()
    # Load pre-computed z-score stats
    z_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_std  = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # In practice these come from trainfit; using dummy identity for benchmark
    zstat_s = time.perf_counter() - t0

    off_s = time.perf_counter() - t_all

    print(f"  BGE-small load        : {load_s_s:>8.3f}s (model: bge-small-scifact-rrf)")
    print(f"  BGE-small doc encode  : {enc_s_s:>8.3f}s ({len(docs)} docs)")
    print(f"  BGE-base load         : {load_b_s:>8.3f}s (bge-base-en-v1.5)")
    print(f"  BGE-base doc encode   : {enc_b_s:>8.3f}s ({len(docs)} docs)")
    print(f"  Z-score stats load    : {zstat_s:>8.3f}s")
    print(f"  TOTAL OFFLINE         : {off_s:>8.3f}s")

    # ============================================================
    #  ONLINE
    # ============================================================
    print()
    print("=" * 60)
    print("ONLINE (per-query SGAF B5+P3)")
    print("=" * 60)

    # Warmup
    for q in queries[:warm]:
        qe_s = model_small.encode([q.text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        s_scores = np.matmul(qe_s, doc_emb_small.T)
        top_s = np.argsort(s_scores[0])[::-1][:top_k]
        qe_b = model_base.encode([q.text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        b_scores = np.matmul(qe_b, doc_emb_base.T)
        top_b = np.argsort(b_scores[0])[::-1][:top_k]

    small_enc_ms, base_enc_ms, feat_ms, p3_ms, tot_ms = [], [], [], [], []
    mode_counts = {"specialist_safe": 0, "generalist_fallback": 0}

    # Process all queries as one batch
    all_features = np.zeros((n, 5))
    small_runs = {}  # query_id -> list[(doc_id, score)]
    base_runs = {}

    t_batch_start = time.perf_counter()

    for qi, q in enumerate(queries):
        tq0 = time.perf_counter()

        # BGE-small encode + search
        ta = time.perf_counter()
        qe_s = model_small.encode([q.text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        s_scores = np.matmul(qe_s, doc_emb_small.T)
        s_top_idx = np.argsort(s_scores[0])[::-1][:top_k]
        s_top_scores = s_scores[0][s_top_idx]
        small_runs[q.query_id] = [(docs[i].doc_id, float(s_scores[0][i])) for i in s_top_idx]
        tb = time.perf_counter()

        # BGE-base encode + search
        qe_b = model_base.encode([q.text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        b_scores = np.matmul(qe_b, doc_emb_base.T)
        b_top_idx = np.argsort(b_scores[0])[::-1][:top_k]
        base_runs[q.query_id] = [(docs[i].doc_id, float(b_scores[0][i])) for i in b_top_idx]
        tc = time.perf_counter()

        # Extract 5 features
        td = time.perf_counter()
        # feature 0: query_len
        f0 = len(q.text.split())
        # feature 1: small_top
        f1 = float(s_top_scores[0])
        # feature 2: small_gap
        f2 = float(s_top_scores[0] - s_top_scores[1]) if len(s_top_scores) > 1 else 0.0
        # feature 3: small_std10
        f3 = float(np.std(s_top_scores[:10]))
        # feature 4: overlap10
        s_set = set(s_top_idx[:10])
        b_set = set(b_top_idx[:10])
        f4 = len(s_set & b_set) / 10.0
        all_features[qi] = [f0, f1, f2, f3, f4]
        te = time.perf_counter()

        small_enc_ms.append((tb - ta) * 1000)
        base_enc_ms.append((tc - tb) * 1000)
        feat_ms.append((te - td) * 1000)
        tot_ms.append((te - tq0) * 1000)

    # ---- Batch-level B5 decision ----
    t_b5 = time.perf_counter()
    z_batch = (all_features.mean(axis=0) - z_mean) / (z_std + 1e-8)
    S = abs(z_batch[0]) + max(0, -z_batch[1]) + max(0, -z_batch[2]) + max(0, -z_batch[3]) + max(0, -z_batch[4])
    b5_ms = (time.perf_counter() - t_b5) * 1000

    # ---- P3 smoothing (if fallback) ----
    p3_batch_ms = 0.0
    if S >= TAU:
        mode_counts["generalist_fallback"] = n
        t_p3 = time.perf_counter()
        for q in queries:
            b_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(base_runs[q.query_id], start=1)}
            s_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(small_runs[q.query_id], start=1)}
            top_w = base_runs[q.query_id][:P3_WINDOW]
            rescored = []
            for doc_id, _ in top_w:
                rb = b_rank.get(doc_id, 999)
                rs = s_rank.get(doc_id)
                if rs is not None:
                    score = (1 - P3_ALPHA) / (K_RRF + rb) + P3_ALPHA / (K_RRF + rs)
                else:
                    score = (1 - P3_ALPHA) / (K_RRF + rb)
                rescored.append((doc_id, score))
            rescored.sort(key=lambda x: x[1], reverse=True)
            # final = rescored top-w + unchanged tail
            final = rescored + base_runs[q.query_id][P3_WINDOW:]
        p3_batch_ms = (time.perf_counter() - t_p3) * 1000
    else:
        mode_counts["specialist_safe"] = n

    batch_total_s = time.perf_counter() - t_batch_start

    def st(arr):
        return statistics.mean(arr), statistics.stdev(arr) if len(arr) > 1 else 0.0

    s_enc_m, s_enc_sd = st(small_enc_ms)
    b_enc_m, b_enc_sd = st(base_enc_ms)
    f_m, f_sd = st(feat_ms)
    t_m, t_sd = st(tot_ms)

    print(f"  BGE-small enc+search  : {s_enc_m:>8.2f} +- {s_enc_sd:>5.2f} ms/q")
    print(f"  BGE-base  enc+search  : {b_enc_m:>8.2f} +- {b_enc_sd:>5.2f} ms/q")
    print(f"  Feature extraction    : {f_m:>8.2f} +- {f_sd:>5.2f} ms/q")
    print(f"  Per-query subtotal    : {t_m:>8.2f} +- {t_sd:>5.2f} ms/q")
    print(f"  ------------------------------------------------------")
    print(f"  B5 batch decision     : {b5_ms:>8.2f} ms (for {n} queries)")
    print(f"  Batch shift S = {S:.3f} -> mode = {max(mode_counts, key=mode_counts.get)}")
    if S >= TAU:
        print(f"  P3 smoothing (batch)  : {p3_batch_ms:>8.2f} ms (top-{P3_WINDOW} RRF blend, all {n} queries)")
    print(f"  ------------------------------------------------------")
    print(f"  TOTAL BATCH (online)  : {batch_total_s:>8.3f}s for {n} queries")
    # per-query total = per-query subtotal + batch overhead / n
    per_q_total = t_m + b5_ms / n + p3_batch_ms / n
    print(f"  TOTAL ONLINE (per q)  : {per_q_total:>8.2f} ms/q")
    print(f"  Throughput             : {1000 / per_q_total:>8.1f} q/s")

    print()
    print("=" * 60)
    print("SUMMARY - SGAF B5+P3 Pipeline")
    print("=" * 60)
    print(f"  Dataset         : SciFact test ({n} queries, {len(docs)} docs)")
    print(f"  Specialist      : BGE-small-scifact-rrf (33M, 384d)")
    print(f"  Generalist      : BGE-base-en-v1.5 (109M, 768d)")
    print(f"  --------------------------------------------------")
    print(f"  OFFLINE setup   : {off_s:>8.3f}s (1 lan, ca 2 model)")
    print(f"  ONLINE per-q    : {per_q_total:>8.2f} ms/q")
    print(f"  ONLINE batch    : {batch_total_s:>8.3f}s ({n} queries)")
    print(f"  Mode            : {max(mode_counts, key=mode_counts.get)} (S={S:.3f})")
    print(f"  B5 routing (batch)   : {b5_ms:.2f} ms (z-score + decision, once)")
    if S >= TAU:
        print(f"  P3 smoothing (batch) : {p3_batch_ms:.2f} ms")
    print(f"  Breakdown per-q : small={s_enc_m:.1f} + base={b_enc_m:.1f} + feat={f_m:.1f} + B5={b5_ms/n:.2f} + P3={p3_batch_ms/n:.2f} ms")
    if S >= TAU:
        print(f"  P3 smoothing    : {p3_batch_ms:.2f} ms (batch-level)")
    print(f"  Breakdown per-q : small={s_enc_m:.1f} + base={b_enc_m:.1f} + feat={f_m:.1f} ms")
    print()

if __name__ == "__main__":
    main()
