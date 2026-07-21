---
title: "Always-Rerank"
type: concept
tags: [baseline, reranking, reference-point]
sources: [seg-research-paper, phase3-progress-2026-06-20, validation-experiments]
last_updated: 2026-06-29
---

## Summary
Always-Rerank is the static baseline that reranks every query's Hybrid RRF top-20 candidates with the cross-encoder. It serves as the high-cost, high-quality endpoint of the effectiveness-vs-cost curve against which selective reranking is compared.

## Performance on SciFact
- nDCG@10: 0.6939 (+0.0356 over Hybrid RRF)
- Recall@10: 0.8286
- Recall@100: 0.9560 (unchanged — only reorders top-20)
- MRR@10: 0.6604
- Rerank coverage: 1.00 (100%)
- Latency: 27.3ms/query GPU

## Role
- Reference point for [[Selective Reranking]] cost-quality trade-off
- Nonconformity score target for [[Conformal Risk Control]]
- Upper bound that mixed BM25/Dense routing cannot reach (Oracle Router 0.7617 still above)
- On k=5 base: nDCG@10=0.7321

## Related To
- [[CrossEncoder]] — the reranker used
- [[Selective Reranking]] — cost-saving alternative
- [[Conformal Risk Control]] — threshold calibrated against this
- [[Reciprocal-Rank-Fusion]] — candidate generator
