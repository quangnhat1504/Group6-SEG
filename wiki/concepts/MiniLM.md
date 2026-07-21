---
title: "MiniLM"
type: concept
tags: [model, lightweight, transformer, cross-encoder]
sources: [seg-research-paper, phase3-progress-2026-06-20]
last_updated: 2026-06-29
---

## Summary
MiniLM is a lightweight transformer model derived through self-attention distillation. In SEG, `cross-encoder/ms-marco-MiniLM-L-6-v2` is used as the cross-encoder reranker and `sentence-transformers/all-MiniLM-L6-v2` was tested as a dense retriever alternative.

## Roles in SEG
- **Cross-encoder reranker**: `ms-marco-MiniLM-L-6-v2`, 27ms/query GPU, produces Always-Rerank nDCG@10=0.6939
- **Dense retriever (ablation)**: nDCG@10=0.6451, competitive standalone but weaker hybrid fusion (0.4683) vs SciNCL
- Chosen for student hardware constraints — fast, lightweight, fits 16GB VRAM

## Related To
- [[CrossEncoder]] — reranking application
- [[Dense Retrieval]] — embedding application
- [[SciNCL]] — preferred scientific dense model
- [[Selective Reranking]] — the system that gates it
