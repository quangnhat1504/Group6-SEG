---
title: "Query Routing"
type: concept
tags: [routing, cost-aware, per-query-selection]
sources: [tasks, phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
Query routing selects the best retrieval method (BM25, Dense, or Hybrid) per query before returning results. This trades off between cheap and expensive retrieval methods based on query characteristics.

## Implemented Routers
- Random Router: baseline, nDCG@10=0.6290
- Majority Router: always BM25, nDCG@10=0.6523 (accuracy 0.7533)
- Oracle Router: upper bound, nDCG@10=0.7617
- Classical TF-IDF LogReg: in-split diagnostic only, nDCG@10=0.7617
- Small LLM QLoRA LogProb: nDCG@10=0.6372
- Small LLM Calibrated Held-Out: nDCG@10=0.6674

## Key Findings
- Oracle shows large headroom (+0.1034 over Hybrid RRF)
- Majority Router hard to beat due to test label BM25-heaviness
- LLM router improves with label log-prob scoring and confidence gating
- Train/test distribution shift is major challenge
- [[Confidence-Gated Fallback]] improves held-out nDCG@10

## Related To
- [[Oracle Router]] — upper bound
- [[QLoRA]] — fine-tuning method for LLM router
- [[Qwen2.5-0.5B-Instruct]] — LLM backbone
- [[Distribution Shift]] — train/test oracle label mismatch
- [[Selective Reranking]] — alternative to routing as cost-saving mechanism
