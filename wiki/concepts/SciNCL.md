---
title: "SciNCL"
type: concept
tags: [dense-model, embedding, scientific, sentence-transformers]
sources: [phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
SciNCL (Neighborhood Contrastive Learning for Scientific Document Representations) is the primary dense retriever model in SEG. It's a SentenceTransformer model (`malteos/scincl`) trained with citation-graph neighborhood contrastive learning for scientific documents.

## Performance on SciFact
- nDCG@10: 0.5640
- Recall@10: 0.7233
- Recall@100: 0.9082 (best among dense models)
- Mrr@10: 0.5224

## Role in SEG Pipeline
- Primary dense component of [[Reciprocal Rank Fusion]] hybrid
- Produces strongest hybrid recall profile (Hybrid RRF Recall@100=0.9560)
- Chosen over SPECTER (0.3523) and MiniLM (weaker hybrid)

## Related To
- [[Dense Retrieval]] — retrieval approach
- [[Reciprocal Rank Fusion]] — hybrid combination
- [[SPECTER]] — failed alternative
- [[MiniLM]] — general-domain alternative
- SentenceTransformers — framework
