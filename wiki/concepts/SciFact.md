---
title: "SciFact"
type: concept
tags: [dataset, benchmark, BEIR, scientific-claims]
sources: [tasks, phase1-phase2-report, seg-research-paper, validation-experiments]
last_updated: 2026-07-11
---

## Summary
SciFact is a scientific claim verification benchmark packaged in BEIR format for retrieval evaluation. The SEG project uses it as the primary dataset. It consists of 5,183 document abstracts, 809 train queries, and 300 test queries. Documents are paper titles and abstracts only (no full-text PDF). The task involves retrieving evidence abstracts for scientific claims, with verification labels SUPPORTS/REFUTES/NEI available for downstream evaluation but not needed for the retrieval task itself.

## Key Properties
- Train: 809 queries (Dense-heavy oracle: BM25=195, Dense=477, Hybrid=137)
- Test: 300 queries (BM25-heavy oracle: BM25=226, Dense=49, Hybrid=25)
- Corpus: 5,183 title+abstract documents
- Significant train/test distribution shift in oracle labels
- Downstream task: scientific claim verification

## Related To
- [[BEIR]] — benchmark framework
- [[BM25 Retrieval]] — strong lexical baseline
- [[SciNCL]] — best dense retriever for this dataset
- [[Reciprocal Rank Fusion]] — hybrid combination
