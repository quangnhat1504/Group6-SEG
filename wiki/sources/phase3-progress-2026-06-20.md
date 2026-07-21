---
title: "Phase 3 Progress — 2026-06-20"
type: source
tags: [progress-report, phase3, always-rerank, gpu]
date: 2026-06-20
source_file: raw/phase3-progress-2026-06-20.md
---

## Summary
Short update marking the completion of the Always-Rerank baseline for Phase 3 selective reranking. CrossEncoder `ms-marco-MiniLM-L-6-v2` reranks Hybrid top-20 candidates on GPU (RTX 5070 Ti), achieving nDCG@10=0.6939 at 27ms/query — a 14x speedup over CPU. Code updated for `--rerank-all` mode and GPU device detection.

## Key Claims
- Always-Rerank baseline completed: nDCG@10=0.6939, MRR@10=0.6604, Recall@10=0.8286
- GPU acceleration: 27ms/query vs 405ms/query CPU (14x speedup)
- Phase 3.1 done; Phase 3.2 (uncertainty triggers) already implemented; 3.3-3.5 pending
- Remaining: threshold ablation, Pareto efficiency plot, final Table 3

## Connections
- [[Always-Rerank]] — completed baseline
- [[CrossEncoder]] — MiniLM-L-6-v2 reranker
- [[Selective Reranking]] — next steps
- [[Reciprocal-Rank-Fusion]] — base retriever for candidate generation
