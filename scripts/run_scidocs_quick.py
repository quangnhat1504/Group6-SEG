"""SciDocs pipeline — BM25 + BGE-base + SciNCL + RRF + Adaptive RRF + CE."""

from __future__ import annotations
import os, sys, time, math
from collections import defaultdict, Counter
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import _bootstrap
import numpy as np

from seg_retrieval.datasets import load_beir_split_direct
from seg_retrieval.fusion import reciprocal_rank_fusion
from seg_retrieval.io import save_run
from seg_retrieval.metrics import evaluate_run, ndcg_at_k
from seg_retrieval.retrievers import BM25Retriever, DenseRetriever, tokenize
from seg_retrieval.rerank import CrossEncoderReranker

RUN_DIR = Path("runs/scidocs")
RUN_DIR.mkdir(parents=True, exist_ok=True)

def stem(w):
    w = w.lower()
    if w.endswith("ies") and len(w) > 3: w = w[:-3] + "y"
    elif w.endswith("es") and len(w) > 3: w = w[:-2]
    elif w.endswith("s") and "ss" not in w[-3:] and len(w) > 2: w = w[:-1]
    if w.endswith("ing") and len(w) > 5: w = w[:-3]
    if w.endswith("ed") and len(w) > 4: w = w[:-2]
    if w.endswith("tion") and len(w) > 5: w = w[:-4] + "t"
    return w

def build_idf(docs, min_df=2):
    N = len(docs); df = Counter()
    for d in docs:
        for t in set(tokenize(d.text)):
            s = stem(t)
            if len(s) >= 3: df[s] += 1
    return {t: math.log(N/c) for t, c in df.items() if c >= min_df}

def q_idf(text, idf):
    stems = [stem(t) for t in tokenize(text) if len(t) >= 3]
    if not stems: return 0.0
    return sum(idf.get(s, 0.0) for s in stems) / len(stems)

def arf(bm25, dense, queries, idf, scale=1.0, max_w=0.9, k=60, top_k=100):
    idfs = [q_idf(q.text, idf) for q in queries]
    center = np.median(idfs)
    fused = {}
    for q in queries:
        midf = q_idf(q.text, idf)
        w_b = 0.05 + (1.0/(1.0 + math.exp(-scale*(midf-center))))*(max_w-0.05)
        scores = defaultdict(float)
        for r, (did, _) in enumerate(bm25.get(q.query_id, []), 1):
            scores[did] += w_b/(k+r)
        for r, (did, _) in enumerate(dense.get(q.query_id, []), 1):
            scores[did] += 1.0/(k+r)
        fused[q.query_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return fused


def main():
    docs, queries, qrels = load_beir_split_direct("data/scidocs", "test")
    print(f"Loaded: {len(docs)} docs, {len(queries)} queries, {len(qrels)} judged")
    lookup = {d.doc_id: d for d in docs}

    # 1. BM25
    print("\n[1] BM25...", flush=True)
    t0 = time.perf_counter()
    bm25_run = BM25Retriever(docs).search(queries, 100)
    save_run(RUN_DIR / "test_bm25.csv", bm25_run)
    bm = evaluate_run(bm25_run, qrels, include_map=True)
    print(f"  nDCG@10={bm['ndcg@10']:.4f} MAP@10={bm['map@10']:.4f} R@10={bm['recall@10']:.4f} R@100={bm['recall@100']:.4f} ({time.perf_counter()-t0:.0f}s)")

    # 2. BGE-base
    print("\n[2] BGE-base...", flush=True)
    t0 = time.perf_counter()
    bge_run = DenseRetriever(docs, "BAAI/bge-base-en-v1.5", 32).search(queries, 100)
    save_run(RUN_DIR / "test_dense_bge_base.csv", bge_run)
    bg = evaluate_run(bge_run, qrels, include_map=True)
    print(f"  nDCG@10={bg['ndcg@10']:.4f} MAP@10={bg['map@10']:.4f} R@10={bg['recall@10']:.4f} R@100={bg['recall@100']:.4f} ({time.perf_counter()-t0:.0f}s)")

    # 3. SciNCL
    print("\n[3] SciNCL...", flush=True)
    t0 = time.perf_counter()
    sc_run = DenseRetriever(docs, "malteos/scincl", 32).search(queries, 100)
    save_run(RUN_DIR / "test_dense_scincl.csv", sc_run)
    sc = evaluate_run(sc_run, qrels, include_map=True)
    print(f"  nDCG@10={sc['ndcg@10']:.4f} MAP@10={sc['map@10']:.4f} R@10={sc['recall@10']:.4f} R@100={sc['recall@100']:.4f} ({time.perf_counter()-t0:.0f}s)")

    # 4. RRF (BGE + BM25)
    print("\n[4] Hybrid RRF...", flush=True)
    rrf_run = reciprocal_rank_fusion([bm25_run, bge_run], k=60, top_k=100)
    save_run(RUN_DIR / "test_hybrid_rrf.csv", rrf_run)
    rr = evaluate_run(rrf_run, qrels, include_map=True)

    rrf_sc = reciprocal_rank_fusion([bm25_run, sc_run], k=60, top_k=100)
    save_run(RUN_DIR / "test_hybrid_scincl.csv", rrf_sc)
    rsc = evaluate_run(rrf_sc, qrels, include_map=True)
    print(f"  BGE RRF: nDCG@10={rr['ndcg@10']:.4f}  SciNCL RRF: nDCG@10={rsc['ndcg@10']:.4f}")

    # 5. Adaptive RRF
    print("\n[5] Adaptive RRF...", flush=True)
    idf = build_idf(docs)
    print(f"  IDF vocab: {len(idf)} terms")
    arf_run = arf(bm25_run, bge_run, queries, idf)
    save_run(RUN_DIR / "test_hybrid_arf.csv", arf_run)
    af = evaluate_run(arf_run, qrels, include_map=True)
    print(f"  nDCG@10={af['ndcg@10']:.4f} MAP@10={af['map@10']:.4f}")

    # 6. CE Rerank
    print("\n[6] CE Rerank...", flush=True)
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce_arf_run, ce_sc_run = {}, {}
    t0 = time.perf_counter()
    for i, q in enumerate(queries, 1):
        ce_arf_run[q.query_id] = reranker.rerank(q, lookup, arf_run.get(q.query_id, []), 20)
        ce_sc_run[q.query_id] = reranker.rerank(q, lookup, rrf_sc.get(q.query_id, []), 20)
        if i % 200 == 0: print(f"  {i}/{len(queries)}...", flush=True)
    el = time.perf_counter() - t0
    save_run(RUN_DIR / "test_always_rerank_arf.csv", ce_arf_run)
    save_run(RUN_DIR / "test_always_rerank_scincl.csv", ce_sc_run)
    cea = evaluate_run(ce_arf_run, qrels, include_map=True)
    ces = evaluate_run(ce_sc_run, qrels, include_map=True)
    print(f"  CE on ARF: nDCG@10={cea['ndcg@10']:.4f}  CE on SciNCL: nDCG@10={ces['ndcg@10']:.4f}  ({el/len(queries)*1000:.0f}ms/q)")

    # Summary
    print("\n" + "=" * 60)
    for name, m in [("BM25", bm), ("SciNCL", sc), ("BGE-base", bg),
                     ("Hybrid RRF", rr), ("Adaptive RRF", af), ("CE on ARF", cea)]:
        print(f"  {name:<18} nDCG@10={m['ndcg@10']:.4f}  MAP@10={m['map@10']:.4f}  "
              f"R@10={m['recall@10']:.4f}  R@100={m['recall@100']:.4f}")


if __name__ == "__main__":
    main()
