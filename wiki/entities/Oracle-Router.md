---
title: "Oracle Router"
type: concept
tags: [upper-bound, routing, reference-point]
sources: [phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
The Oracle Router uses ground-truth per-query best-route labels (highest nDCG@10 among BM25/Dense/Hybrid) to always select the optimal retrieval method. It is an upper bound, not a deployable system.

## Performance on SciFact
- Accuracy: 1.0000, Macro-F1: 1.0000
- nDCG@10: 0.7617 (+0.1034 over Hybrid RRF)
- Recall@10: 0.8711
- MRR@10: 0.7337
- Shows large headroom for routing

## Oracle Label Distribution
- Test: BM25=226, Dense=49, Hybrid=25 (BM25-heavy)
- Train: BM25=195, Dense=477, Hybrid=137 (Dense-heavy)
- Distribution shift makes LLM router training difficult

## Related To
- [[Query Routing]] — the task
- [[Confidence-Gated Fallback]] — practical alternative
- [[Distribution Shift]] — train/test mismatch
