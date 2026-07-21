---
title: "SEG Validation Experiments Report"
type: source
tags: [validation, statistical-tests, cross-dataset, nfcorpus]
date: 2026-06-26
source_file: raw/validation-experiments.md
---

## Summary
Six validation experiments strengthening the SEG paper: (1) statistical significance of conformal vs always-rerank (not significant, p=0.148 — ideal for cost-saving narrative); (2) QPP feature selection verified leak-free (hybrid_max #1 on both train/test); (3) CRC guarantees hold on NFCorpus cross-dataset; (4) Selective reranking beats always-rerank on stronger k=5 base; (5) Top-50 reranking provides no improvement over top-20 while doubling latency; (6) Positioning vs Al-Joofi et al. documented with protocol differences.

## Key Claims
- Selective reranking maintains comparable nDCG at 61% coverage (not significantly different from always-rerank)
- hybrid_max is the top QPP predictor on both train (|τ|=0.184) and test (|τ|=0.166) — no leakage
- CRC guarantees hold on NFCorpus (risk=0.012 ≤ α=0.02)
- Selective reranking on k=5 base: 0.7408 vs Always-Rerank 0.7321
- Top-50 provides zero gain over top-20 with 92% slower latency
- Direct comparison with Al-Joofi invalid due to protocol differences (300 vs 100 queries)

## Connections
- [[Conformal Risk Control]] — Experiments 1, 3, 4
- [[Query Performance Prediction]] — Experiment 2
- [[NFCorpus]] — Experiment 3 cross-dataset
- [[Reciprocal Rank Fusion]] — Experiment 4 k=5 variant
- [[Always-Rerank]] — baseline for all comparisons
- [[Selective Reranking]] — the approach validated
- [[Al-Joofi-et-al]] — related work positioning
