---
title: "Adaptive RRF"
type: concept
tags: [adaptive-rrf, fusion, idf, bm25, dense]
last_updated: 2026-07-11
---

## What

Adaptive RRF is a per-query weighted reciprocal rank fusion method that adjusts BM25 weight based on query IDF statistics. Queries with rare technical terms (high IDF) get higher BM25 weight for exact matching; queries with common vocabulary (low IDF) lean on dense semantic matching.

- **Inspiration**: vstash (Steffens 2026, arXiv 2604.15484)
- **Cost**: Zero-shot (no training), only corpus IDF stats needed
- **Key formula**: w_bm25 = sigmoid(γ × (mean_IDF_query - c)), scaled to [0.05, 0.95]

## How It Works

1. Build IDF vocabulary from full corpus: IDF(t) = log(N / df(t))
2. For each query: compute mean IDF of stemmed query terms
3. Pass through sigmoid: w_bm25 = f(mean_IDF), w_dense = 1.0
4. RRF fuse with per-query weights

## Results

| Dataset | BGE-base MAP@10 | Adaptive RRF MAP@10 | Δ | p-value |
|---------|-----------------|---------------------|---|---------|
| SciFact | 0.6918 | 0.7111 | +0.0192 | 0.085 (n.s.) |
| NFCorpus | 0.2773 | 0.2778 | +0.0004 | 0.908 (n.s.) |
| FiQA | 0.3135 | 0.3007 | −0.0127 | 0.083 (n.s.) |

**Pattern**: Adaptive RRF helps when BM25 is strong (SciFact, BM25 MAP@10 = 0.6071). It adds noise when BM25 is weak (FiQA, BM25 MAP@10 = 0.1600). The improvements are **not statistically significant** on any dataset — useful as a technical upgrade but the quantitative gain is modest.

## Parameters

- γ = 5.0 (sigmoid steepness)
- w_min = 0.05, w_max = 0.95
- Center = median query mean_IDF on train split
- k = 60 (RRF constant)

## Related

- [[Reciprocal Rank Fusion]] — equal-weight RRF (baseline)
- [[BGE]] — dense retriever used for fusion
- [[BM25 Retrieval]] — sparse retriever used for fusion
- [[Paired Bootstrap]] — significance testing
