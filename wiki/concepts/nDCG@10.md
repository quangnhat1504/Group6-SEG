---
title: "nDCG@10"
type: concept
tags: [metrics, evaluation, ndcg]
last_updated: 2026-07-11
---

## What

**Normalized Discounted Cumulative Gain at 10** — the primary evaluation metric in this project, following [[Al-Joofi-et-al]] (Applied Sciences, 2025) who use nDCG@10 as their primary comparison measure.

## Definition

For a single query:
- DCG@10 = Σ(2^rel_i - 1) / log₂(i + 1) for i=1..10
- IDCG@10 = DCG of ideal ranking (documents sorted by true relevance)
- nDCG@10 = DCG@10 / IDCG@10

nDCG@10 = mean of per-query nDCG@10 across all queries

## Why nDCG@10 over MAP@10

- **Graded relevance**: SciFact uses relevance scores 0/1/2, not just binary. nDCG captures the magnitude of relevance.
- **Top-weighted**: Logarithmic discount gives more weight to top-ranked documents.
- **Al-Joofi 2025**: Uses nDCG@10 as the primary comparison measure across all their experiments.

## Usage in SEG

All paper tables report nDCG@10 as the primary metric, with MAP@10, Recall, and MRR as supplementary. Significance tested via paired t-test with Shapiro-Wilk normality check.

## Key Results (nDCG@10 on SciFact test, 300 queries)

| Method | nDCG@10 |
|--------|---------|
| BM25 | 0.6523 |
| SciNCL (old) | 0.5640 |
| BGE-base | 0.7376 |
| Adaptive RRF | 0.7514 |
| CE on ARF | 0.7140 |

## Related

- [[MAP@10]] — secondary metric (binary relevance)
- [[Paired-Bootstrap]] — significance testing on nDCG@10
- [[BGE]] — achieves nDCG@10 = 0.7376 on SciFact
- [[Adaptive RRF]] — pushes nDCG@10 to 0.7514
