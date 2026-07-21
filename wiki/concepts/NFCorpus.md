---
title: "NFCorpus"
type: concept
tags: [dataset, cross-dataset, biomedical, BEIR]
sources: [seg-research-deep-plan, validation-experiments]
last_updated: 2026-07-11
---

## Summary
NFCorpus is a biomedical BEIR dataset used as the first cross-dataset validation target for the SEG pipeline. 3,633 documents, 323 test queries. Also known as TREC Precision Medicine or TREC-COVID predecessor.

## SEG Results
- BM25: nDCG@10=0.1915
- Dense/SciNCL: nDCG@10=0.2252
- Hybrid RRF k=60: nDCG@10=0.2591
- Always-Rerank: nDCG@10=0.3226
- Best QPP signal: bm25_std (τ=+0.151) — differs from SciFact's hybrid_max
- CRC guarantee holds: coverage=79%, realized risk=0.012 ≤ α=0.02

## Key Findings
- Pipeline architecture generalizes to different biomedical domain
- QPP signal identity is dataset-specific
- CRC guarantee mechanism works regardless of which signal is used
- Cross-encoder reranking is largest effectiveness driver on both datasets

## Related To
- [[SciFact]] — primary dataset
- [[BEIR]] — benchmark framework
- [[Conformal Risk Control]] — guarantee mechanism
- [[Query Performance Prediction]] — signal differs from SciFact
