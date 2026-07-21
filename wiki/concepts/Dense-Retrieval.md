---
title: "Dense Retrieval"
type: concept
tags: [semantic, neural, embedding, retrieval]
sources: [tasks, phase1-phase2-report, seg-research-paper]
last_updated: 2026-07-11
---

## Summary
Dense retrieval uses SentenceTransformers semantic embeddings to rank documents by vector similarity. The primary dense model is [[SciNCL]]. Documents are encoded once and cached; queries are encoded at search time and compared via dot-product.

## Performance on SciFact
- SciNCL: nDCG@10=0.5640, Recall@10=0.7233, Recall@100=0.9082, MRR@10=0.5224
- SPECTER: nDCG@10=0.3523 (poor, failed ablation)
- MiniLM: nDCG@10=0.6451 (competitive but weaker hybrid)

## Role in SEG Pipeline
- Adds semantic coverage beyond exact term matching
- Better Recall@100 than BM25 but weaker top-10 precision
- Critical component of [[Reciprocal Rank Fusion]] hybrid
- Test oracle: best route for 49/300 queries

## Related To
- [[BM25 Retrieval]] — complementary lexical approach
- [[SciNCL]] — scientific embedding model
- [[SPECTER]] — failed alternative
- [[MiniLM]] — general-domain alternative
- [[Reciprocal Rank Fusion]] — hybrid combination
