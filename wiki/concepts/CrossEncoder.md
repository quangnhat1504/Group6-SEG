---
title: "CrossEncoder"
type: concept
tags: [reranker, cross-encoder, relevance-scoring]
sources: [seg-research-paper, phase3-progress-2026-06-20, validation-experiments]
last_updated: 2026-06-29
---

## Summary
Cross-encoders jointly encode query-document pairs and produce relevance scores. SEG uses the lightweight `cross-encoder/ms-marco-MiniLM-L-6-v2` model as the expensive reranking stage that is gated by uncertainty signals.

## Performance
- Always-Rerank: nDCG@10=0.6939 (+0.0356 over Hybrid RRF)
- GPU latency: 27.3ms/query (RTX 5070 Ti)
- CPU latency: ~405ms/query
- Top-20 candidate reranking depth (top-50 provides no gain)
- Produces monotonically better rankings (no new documents introduced)
- Top-50 doubled latency with zero gain (Δ=−0.0023, p=0.68)

## Role in SEG Pipeline
- The expensive stage that [[Selective Reranking]] gates
- Always-Rerank serves as upper-bound reference for [[Conformal Risk Control]]
- Purpose-built for relevance — far better than [[Downstream-Utility Reranking]]
- Runs on GPU with auto-detection, falls back to CPU

## Related To
- [[Selective Reranking]] — gating mechanism
- [[Always-Rerank]] — baseline reranking all queries
- [[Conformal Risk Control]] — threshold selection
- [[MiniLM]] — model architecture
- [[Reciprocal-Rank-Fusion]] — candidate generator
