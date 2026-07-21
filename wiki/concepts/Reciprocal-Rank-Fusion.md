---
title: "Reciprocal Rank Fusion"
type: concept
tags: [fusion, hybrid, rank-aggregation, retrieval]
sources: [tasks, phase1-phase2-report, seg-research-paper, validation-experiments]
last_updated: 2026-07-11
---

## Summary
RRF combines multiple ranked lists into a single ranking using `1/(k + rank)`. In SEG, it fuses BM25 and Dense/SciNCL results with rrf_k=60 (default) or k=5 (stronger, amplifies top ranks).

## Performance on SciFact
- k=60: nDCG@10=0.6583, Recall@10=0.8146, Recall@100=0.9560, MRR@10=0.6157
- k=5: nDCG@10=0.6809 (+0.0226 over k=60)
- Best Recall@100 among all base retrievers
- Test oracle: best route for 25/300 queries

## Design Decisions
- k=60 is the default, not tuned on test
- k=5 confirmed as stronger in validation (Experiment 4)
- Top-100 candidate set passed to reranking stage
- Hybrid max RRF score is the best QPP signal (Kendall +0.499 vs base nDCG)

## Related To
- [[BM25 Retrieval]] — lexical component
- [[Dense Retrieval]] — semantic component
- [[SciNCL]] — dense model
- [[Selective Reranking]] — candidates from Hybrid
- [[Query Performance Prediction]] — hybrid_max as top signal
