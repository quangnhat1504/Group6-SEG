---
title: "Conformal Risk Control"
type: concept
tags: [statistical-guarantee, threshold-selection, risk-control]
sources: [seg-research-paper, seg-research-directions, validation-experiments, phase3-progress-2026-06-24]
last_updated: 2026-06-29
---

## Summary
Principled threshold selection method for selective reranking. Instead of grid-search, CRC provides a finite-sample guarantee that expected nDCG shortfall versus Always-Rerank stays within budget α. Based on the Learn-Then-Test framework.

## How It Works in SEG
- Calibration split: 150 queries used to select threshold λ
- Per-query loss: nDCG sacrificed by not reranking (0 if reranked, max(0, rerank_ndcg − base_ndcg) if skipped)
- Loss is monotone in threshold
- Select smallest-coverage λ satisfying (n·empirical_risk + B)/(n+1) ≤ α
- Guarantee is in expectation over calibration draws (B=1 conservative bound)

## Results
- α=0.02: 61% coverage, nDCG@10=0.7280, realized risk=0.0052
- α=0.02 (score-gap): 72% coverage, nDCG@10=0.7120
- QPP signal saves 11 coverage points vs heuristic under same guarantee
- Averaged over 200 splits: 0/38 operating points violate E[risk] ≤ α
- CRC guarantees hold on [[NFCorpus]] cross-dataset (risk=0.012 ≤ α=0.02)

## Related To
- [[Selective Reranking]] — the application
- [[Query Performance Prediction]] — trigger signal
- [[Always-Rerank]] — upper bound reference
- [[CrossEncoder]] — the expensive stage being gated
