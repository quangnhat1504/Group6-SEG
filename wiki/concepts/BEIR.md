---
title: "BEIR"
type: concept
tags: [benchmark, heterogeneous, zero-shot, retrieval]
sources: [tasks, seg-research-paper, validation-experiments]
last_updated: 2026-06-29
---

## Summary
BEIR (Benchmarking IR) is a heterogeneous zero-shot retrieval evaluation benchmark. SEG uses BEIR's SciFact packaging for standardized data loading, evaluation metrics, and retrieval interface. Additional BEIR datasets (NFCorpus, FiQA, ArguAna, TREC-COVID) are candidates for cross-dataset validation.

## Role in SEG
- Provides consistent data format (corpus, queries, qrels)
- Standard evaluation metrics (nDCG@k, Recall@k, MRR@k)
- Allows comparison across diverse domains
- SciFact is the primary BEIR dataset; NFCorpus is validation target

## Related To
- [[SciFact]] — primary dataset
- [[NFCorpus]] — cross-dataset validation
- [[BM25 Retrieval]] — evaluated in BEIR framework
- [[Dense Retrieval]] — evaluated in BEIR framework
