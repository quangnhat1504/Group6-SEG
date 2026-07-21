---
title: "SEG Research Directions"
type: source
tags: [research-planning, future-directions]
date: 2026-06-21
source_file: raw/seg-research-directions.md
---

## Summary
Survey of high-novelty research directions for the SEG project, derived from fan-out web search over 2023-2026 IR/ML literature. Ranks five directions on novelty, feasibility, and coherence with the SEG cost/uncertainty thesis. The primary recommendation is conformal risk-controlled selective reranking (Direction 1), with QPP (Direction 2) as the gating mechanism, and downstream-utility reranking via semantic entropy (Direction 3) as the most innovative extension.

## Key Claims
- Direction 1 (Conformal selective reranking) is the top recommendation: high novelty, perfect thesis alignment, low effort
- Direction 3 (Downstream-utility reranking) is the most publishable: connects retrieval to SciFact's claim verification task
- Direction 2 (QPP as unified gating signal) provides quantitative grounding for trigger decisions
- Five real sources identified for each direction
- Effort estimates: Directions 1&2 are low-medium; Direction 3 is medium; Direction 4 (listwise LLM) is medium-high
- Directions 4 (listwise LLM) and 5 (GPL domain adaptation) are secondary/optional

## Connections
- [[Conformal Risk Control]] — Direction 1 backbone
- [[Query Performance Prediction]] — Direction 2 backbone
- [[Downstream-Utility Reranking]] — Direction 3 backbone
- [[Semantic Entropy]] — LLM uncertainty for reranking
- [[SciFact]] — downstream verification task
- [[Selective Reranking]] — the threshold optimization problem
- [[CrossEncoder]] — current expensive stage
- [[Qwen2.5-0.5B-Instruct]] — small LLM for utility signals
