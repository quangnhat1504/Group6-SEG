---
title: "MAP@10"
type: concept
tags: [metrics, evaluation, map]
last_updated: 2026-07-11
---

## What

**Mean Average Precision at 10** — a secondary evaluation metric in this project. The primary metric is **nDCG@10**, following [[Al-Joofi-et-al]] (Applied Sciences, 2025) who use nDCG@10 as their primary comparison measure.

## Definition

For a single query:
- AP@10 = (1 / min(R, 10)) × Σ(P@k × rel(k)) for k=1..10
- Where R = number of relevant documents, P@k = precision at rank k, rel(k) = 1 if document at rank k is relevant, 0 otherwise

MAP@10 = mean of AP@10 across all queries

## Why MAP@10 ≠ nDCG@10

- MAP@10: Binary relevance (doc is either relevant or not). High rank of non-relevant docs hurts linearly.
- nDCG@10: Graded relevance (0, 1, 2+). Higher relevance scores get higher gain. Penalizes non-relevant docs logarithmically.

## Usage in SEG

nDCG@10 is the primary metric (aligned with Al-Joofi 2025). MAP@10 is reported as a secondary metric in all paper tables. Significance tested via paired t-test with Shapiro-Wilk normality check.

## Related

- [[Paired-Bootstrap]] — significance testing on nDCG@10
- [[BGE]] — achieves nDCG@10 = 0.7376 on SciFact
- [[Adaptive RRF]] — pushes nDCG@10 to 0.7514
- [[SciFact]] — primary evaluation dataset
