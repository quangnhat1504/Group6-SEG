---
title: "SEG Deep Research Plan"
type: source
tags: [research-planning, next-steps, 2-week-sprint]
date: 2026-06-27
source_file: raw/seg-research-deep-plan.md
---

## Summary
Detailed next-direction plan addressing current limitations (weak QPP signal, single dataset, non-significant differences, weak router) with five concrete directions ranked by ROI: (1) Learned QPP regression; (2) Conformal for RAG context selection; (3) Multi-dataset BEIR generalization; (4) Adaptive rerank depth per query; (5) Self-reflective QPP using cross-encoder confidence. Proposes a 2-week sprint with specific day-by-day milestones. Each direction has concrete implementation phases and expected impact estimates.

## Key Claims
- Learned QPP regression could improve Kendall τ from 0.166 to 0.25-0.35 and save ~15% compute
- Multi-dataset validation (FiQA, ArguAna, TREC-COVID) transforms paper from single-dataset to multi-domain
- Cascaded CRC (retrieval → reranking → RAG context) is the most novel extension
- Adaptive depth per query could reduce latency further beyond binary skip/rerank
- Two-stage CRC (QPP cheap signal → CE confidence as quality check) addresses when reranking hurts
- All directions are feasible on a single RTX 5070 Ti within 1-2 weeks

## Connections
- [[Query Performance Prediction]] — Directions 1, 5 core
- [[Conformal Risk Control]] — Directions 2, 4 core
- [[BEIR]] — Direction 3 multi-dataset validation
- [[CrossEncoder]] — Direction 5 self-reflective source
- [[Qwen2.5-0.5B-Instruct]] — RAG generator for Direction 2
- [[SciFact]] — primary dataset
- [[NFCorpus]] — cross-dataset validation target
- [[Selective Reranking]] — the skip/rerank binary current approach
