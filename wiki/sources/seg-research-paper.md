---
title: "SEG Research Paper — Current Progress"
type: source
tags: [progress-report, paper, complete]
date: 2026-06-20
source_file: raw/seg-research-paper.md
---

## Summary
Comprehensive progress report structured as an academic paper covering all SEG phases through Phase 3 selective reranking plus Phase 4 principled trigger upgrades (QPP + Conformal). Presents results from BM25/Dense/Hybrid base retrieval through Always-Rerank baseline (nDCG@10=0.6939), threshold ablation, QPP signals, Conformal Risk Control (61% coverage at α=0.02, nDCG@10=0.7280), and a negative result for downstream-utility reranking. The central thesis: cheap QPP signals + conformal guarantee decide when expensive cross-encoder reranking is worth it.

## Key Claims
- Hybrid RRF (nDCG@10=0.6583) is the best base retriever
- Always-Rerank improves to nDCG@10=0.6939 at 27ms/query on GPU
- Oracle Router shows headroom at nDCG@10=0.7617
- Hybrid max score predicts gain-from-reranking best (Kendall -0.166 vs -0.060 for score gap)
- CRC at α=0.02: 61% coverage, nDCG@10=0.7280, risk guarantee holds
- QPP signal saves 11 coverage points vs heuristic under same guarantee
- Downstream-utility reranking with Qwen2.5-0.5B is a negative result (AUC≈0.64)
- 19 references covering BM25, RRF, SciNCL, QLoRA, Conformal, QPP, and MiniLM

## Connections
- [[SciFact]] — primary evaluation dataset
- [[BM25 Retrieval]] — lexical retriever
- [[Dense Retrieval]] — semantic retriever
- [[Reciprocal Rank Fusion]] — hybrid fusion method
- [[SciNCL]] — scientific dense model
- [[Query Routing]] — Oracle/Classical/LLM routers
- [[Selective Reranking]] — uncertainty-gated cross-encoder
- [[CrossEncoder]] — reranker (MiniLM-L-6-v2)
- [[Always-Rerank]] — static baseline reranking all queries
- [[Conformal Risk Control]] — principled threshold algorithm
- [[Query Performance Prediction]] — unsupervised rerank-trigger signals
- [[Downstream-Utility Reranking]] — experiment with small LLM claim verification
- [[Qwen2.5-0.5B-Instruct]] — small LLM
- [[QLoRA]] — efficient fine-tuning
- [[Semantic Entropy]] — uncertainty in LLM verification
