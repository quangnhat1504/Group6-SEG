---
title: "BM25 Retrieval"
type: concept
tags: [lexical, sparse, baseline, retrieval]
sources: [tasks, phase1-phase2-report, seg-research-paper]
last_updated: 2026-07-11
---

## Summary
BM25 is a classic probabilistic lexical retrieval function used as the sparse baseline in SEG. It ranks documents by exact term overlap between query and document text, weighted by term frequency and inverse document frequency.

## Performance on SciFact
- nDCG@10: 0.6523
- Recall@10: 0.7757
- Recall@100: 0.8731
- MRR@10: 0.6184
- Cost: lowest (1 unit)
- Test oracle: best route for 226/300 queries

## Role in SEG Pipeline
- Serves as the low-cost, high-quality lexical baseline
- Strong on queries with distinctive biomedical/scientific terminology
- One component of [[Reciprocal Rank Fusion]] hybrid
- Majority Router always selects BM25 (accuracy 0.7533)

## Related To
- [[Dense Retrieval]] — complementary semantic approach
- [[Reciprocal Rank Fusion]] — hybrid combination
- [[Selective Reranking]] — reranking candidate results
