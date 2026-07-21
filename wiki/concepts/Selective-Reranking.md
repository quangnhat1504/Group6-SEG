---
title: "Selective Reranking"
type: concept
tags: [reranking, uncertainty, cost-efficiency, cross-encoder]
sources: [tasks, phase1-phase2-report, seg-research-paper, phase3-progress-2026-06-20, validation-experiments]
last_updated: 2026-06-29
---

## Summary
The core SEG contribution: use uncertainty signals to decide per-query whether to invoke an expensive cross-encoder reranker. Queries flagged as uncertain get reranked; queries with confident base results skip reranking. This saves compute while preserving quality.

## Key Results
- Always-Rerank baseline: nDCG@10=0.6939, 27ms/query GPU
- Heuristic selective: recovers 90% of Always-Rerank gain at 66% coverage
- QPP + Conformal: nDCG@10=0.7280 at 61% coverage (α=0.02)
- Selective surpasses Always-Rerank on k=5 base (+0.0088)
- Not significantly different from Always-Rerank (p=0.148) — ideal for cost-saving

## Uncertainty Signals
- Score gap: top-1 vs top-2 retrieval score difference
- Retriever disagreement: 1 − Jaccard overlap between BM25/Dense top-10
- Router confidence: LLM router's label margin
- [[Query Performance Prediction]] signals (hybrid_max, NQC, WIG, std)

## Related To
- [[CrossEncoder]] — the reranker model
- [[Always-Rerank]] — baseline reranking everything
- [[Conformal Risk Control]] — principled threshold selection
- [[Query Performance Prediction]] — trigger signal source
- [[Reciprocal-Rank-Fusion]] — base retriever providing candidates
