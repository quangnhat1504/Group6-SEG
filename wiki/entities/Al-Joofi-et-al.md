---
title: "Al-Joofi et al."
type: entity
tags: [related-work, comparison, scientific-retrieval]
sources: [validation-experiments]
last_updated: 2026-07-11
---

## Summary
Al-Joofi et al. (2025) is a related scientific retrieval study that also applies RRF and cross-encoder reranking to SciFact. Their primary evaluation metric is **nDCG@10**, with paired t-test for significance (Shapiro-Wilk normality check, Wilcoxon fallback).

## Shared Findings (Corroborated)
- Lower RRF k strengthens hybrid base (k-dilution effect)
- Cross-encoder reranking is primary performance driver (on weak dense retrievers)
- Hybrid fusion outperforms individual retrievers

## Protocol Differences
- Test queries: SEG uses 300, Al-Joofi uses 100
- Dense retriever: SEG uses SciNCL, Al-Joofi uses SPECTER/SciBERT
- Rerank depth: SEG uses top-20, Al-Joofi uses top-100
- **Primary metric**: Both use nDCG@10 (SEG aligns with Al-Joofi)
- **Significance test**: Both use paired t-test with Shapiro-Wilk normality check

## SEG's Novel Contributions Beyond Al-Joofi
- QPP-based selective gating with conformal guarantees
- Formal risk control (E[risk] ≤ α) for threshold selection
- LLM-based downstream-utility reranking
- Efficiency analysis showing selective approach saves ~39% compute
- Demonstration that modern embeddings (BGE) make the entire pipeline unnecessary

## Related To
- [[Selective Reranking]] — SEG's distinguishing contribution
- [[Conformal Risk Control]] — unique SEG contribution
- [[Reciprocal Rank Fusion]] — shared method
- [[CrossEncoder]] — shared method
- [[SciFact]] — shared dataset
