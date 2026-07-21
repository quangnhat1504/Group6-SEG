---
title: "June 29 Experiment Session — Pipeline Redesign"
type: source
tags: [experiment-session, final, pipeline-redesign]
date: 2026-06-29
source_file: raw/seg-final-report.md
---

## Summary
Full-day experiment session that completely redesigned the SEG pipeline. The key discovery: replacing SciNCL (0.5640) with BGE-base-en-v1.5 (0.7376) eliminates the need for the entire multi-stage retrieval architecture. Tested 20+ configurations across 3 BEIR datasets. Final best pipeline: BGE-base + Adaptive RRF (0.7514 nDCG@10 on SciFact, closing 90% of oracle gap).

## Experiments Run
- Learned QPP regressor (negative — distribution shift)
- Query expansion with HyDE + RM3 (negative — +1.6% only)
- Asymmetric weighted RRF (modest — +1.2% with SciNCL)
- Dense model ablation (breakthrough — BGE-base at 0.7376)
- Adaptive RRF with per-query IDF weighting (winner — 0.7514)
- Cross-encoder Always-Rerank + CRC on all configurations (negative — always degrades)
- BM25 parameter tuning (negative — distribution shift)
- Full combo pipelines (confirmed optimal configuration)
- Cross-dataset validation on NFCorpus and FiQA (confirmed generalization)

## Key Findings
- BGE-base-en-v1.5 alone beats the entire old pipeline at lower cost
- Cross-encoder reranking always hurts BGE models
- Train/test distribution shift breaks all learned methods
- Adaptive RRF adds small but consistent gains on datasets with strong BM25
- BGE-large shows negative scaling (bigger model underperforms)
- Single model swap closes 76.7% of oracle gap alone; adaptive RRF takes it to 90.0%

## Connections
- [[BGE]] — primary discovery
- [[Adaptive RRF]] — best fusion method
- [[SciNCL]] — retired baseline
- [[Selective Reranking]] — rendered unnecessary
- [[CrossEncoder]] — confirmed harmful
- [[Conformal Risk Control]] — no longer needed
- [[SciFact]] — primary dataset
- [[NFCorpus]] — cross-dataset validation
- [[FiQA]] — cross-dataset validation
- [[Distribution Shift]] — root cause of multiple failures
- [[BGE]] — strong compact alternative
- [[BGE]] — negative scaling example
