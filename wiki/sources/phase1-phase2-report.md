---
title: "Phase 1-2 Progress Report"
type: source
tags: [progress-report, phase1, phase2]
date: 2026-05-26
source_file: raw/phase1-phase2-report.md
---

## Summary
Report covering the completion of Phase 1 (base retrieval) and Phase 2 (query routing). Phase 1 established BM25, Dense/SciNCL, and Hybrid RRF baselines on SciFact. Phase 2 implemented Router baselines (Random, Majority, Oracle), classical TF-IDF Logistic Regression router, and a QLoRA-tuned Small LLM router. Key finding: Hybrid RRF (nDCG@10=0.6583) is the strongest base retriever, and oracle routing shows headroom (nDCG@10=0.7617). Small LLM router is limited by class imbalance and distribution shift but improves with label log-probability scoring and confidence-gated fallback.

## Key Claims
- BM25 (nDCG@10=0.6523) is a strong baseline; Dense/SciNCL (0.5640) better for recall
- Hybrid RRF (nDCG@10=0.6583) gives best Phase 1 result with Recall@100=0.9560
- SPECTER dense model underperforms (nDCG@10=0.3523)
- Test oracle: BM25-heavy (226/49/25); Train oracle: Dense-heavy — major distribution shift
- Classical TF-IDF LogReg router is in-split diagnostic only (nDCG@10=0.7617)
- Small LLM QLoRA LogProb: nDCG@10=0.6372; Calibrated Held-Out: 0.6674
- Confidence-gated fallback to Hybrid improves nDCG@10 on held-out queries

## Connections
- [[SciFact]] — dataset
- [[BM25 Retrieval]] — strongest low-cost baseline
- [[Dense Retrieval]] — semantic baseline
- [[Reciprocal Rank Fusion]] — hybrid combination
- [[SciNCL]] — primary dense model
- [[SPECTER]] — failed dense ablation
- [[MiniLM]] — competitive general-domain alternative
- [[Query Routing]] — per-query method selection
- [[QLoRA]] — fine-tuning method
- [[Qwen2.5-0.5B-Instruct]] — LLM backbone
- [[Oracle Router]] — upper bound for routing
- [[Confidence-Gated Fallback]] — calibrated LLM routing
- [[Distribution Shift]] — train/test oracle label mismatch
