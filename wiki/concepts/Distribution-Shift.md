---
title: "Distribution Shift"
type: concept
tags: [data, train-test, label-shift, router]
sources: [phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
A major challenge for the SEG query router: the train oracle labels are Dense-heavy (BM25=195, Dense=477, Hybrid=137) while test oracle labels are BM25-heavy (226/49/25). This causes the LLM router trained on train to overpredict Dense on test.

## Causes
- Oracle label defined by per-query nDCG@10 — small top-rank changes can flip labels
- SciFact splits may contain different claim types and terminology specificity
- Dense/SciNCL may retrieve broader candidates but be less stable at exact top ranks
- BM25 may dominate test queries with distinctive biomedical terms

## Impact
- LLM router overpredicts Dense on test
- Majority Router (always BM25) is difficult to beat despite being naive
- Motivates [[Confidence-Gated Fallback]] and retrieval-aware features
- Potential conformal exchangeability violation

## Related To
- [[Oracle Router]] — label source
- [[Query Routing]] — affected system
- [[Confidence-Gated Fallback]] — mitigation
- [[SciFact]] — dataset
