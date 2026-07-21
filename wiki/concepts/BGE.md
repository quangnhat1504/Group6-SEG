---
title: "BGE (BAAI General Embedding)"
type: concept
tags: [bge, embedding, dense-retrieval, bert]
last_updated: 2026-07-11
---

## What

**BGE-base-en-v1.5** is a BERT-based general-purpose embedding model from BAAI (Beijing Academy of AI). It maps text documents and queries into a shared 768-dimensional vector space where semantic similarity corresponds to cosine distance.

- **Model**: `BAAI/bge-base-en-v1.5` (109M parameters, 768-dim)
- **Training**: MTEB-style contrastive learning across diverse tasks (retrieval, clustering, classification, STS)
- **Use in SEG**: Replaces SciNCL as primary dense retriever — loads via SentenceTransformers, encodes title+abstract, computes cosine similarity against query embeddings

## Why It Matters

BGE-base is the **key finding** of this project. A single dense retriever using BGE-base (nDCG@10 = 0.7376, MAP@10 = 0.6918 on SciFact) outperforms the entire old multi-stage pipeline (SciNCL + BM25 + RRF + cross-encoder, nDCG@10 = 0.6939, MAP@10 = 0.6448) at lower cost (3 units vs 4+CE) and 99% lower latency (0.3ms vs 28ms).

BGE's strength makes fusion, reranking, and selective gating largely unnecessary — the pipeline paradox: simpler is better when the base model is strong enough.

## Model Variants Tested

| Model | Params | Dims | SciFact nDCG@10 | MAP@10 |
|-------|--------|------|-----------------|--------|
| BGE-small-en-v1.5 | 34M | 384 | 0.7200 | 0.6764 |
| **BGE-base-en-v1.5** | **109M** | **768** | **0.7376** | **0.6918** |
| BGE-large-en-v1.5 | 335M | 1024 | 0.7346 | 0.6910 |

Negative scaling: BGE-large (335M) underperforms BGE-base (109M) on the small SciFact corpus (5,183 docs) — likely overfitting.

## Related

- [[Dense Retrieval]] — general dense retrieval methodology
- [[SciNCL]] — old scientific embedding model (superseded by BGE)
- [[Adaptive RRF]] — per-query IDF-weighted fusion with BM25
- [[nDCG@10]] — primary evaluation metric
- [[Paired-Bootstrap]] — significance testing
