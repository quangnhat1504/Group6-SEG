---
title: "Phase 3 Progress — 2026-06-24 (Direction 3 Complete)"
type: source
tags: [progress-report, phase3, downstream-utility, negative-result]
date: 2026-06-24
source_file: raw/phase3-progress-2026-06-24.md
---

## Summary
Direction 3 (downstream-utility reranking via small LLM semantic entropy) is complete and is a negative result. Qwen2.5-0.5B-Instruct verifier yields weak separation of relevant vs non-relevant abstracts (AUC~0.60-0.64), and reranking never beats the Hybrid base. The 0.5B model is overconfident (non-relevant mean confidence 0.924). This strengthens the thesis: value is in cheap signals deciding when to invoke a strong purpose-built reranker, not in making the expensive stage reason about the downstream task.

## Key Claims
- Downstream-utility reranking complete, written as Section 6.10
- Weak signal separation: AUC~0.60-0.64 for relevant vs non-relevant
- Never beats Hybrid base (nDCG@10=0.6583)
- 0.5B verifier inference: 37ms/pair GPU vs 870ms CPU
- Conformal-gated selective utility reranking lowers nDCG at every coverage
- Conclusion: utility signal not a substitute for purpose-built cross-encoder

## Connections
- [[Downstream-Utility Reranking]] — completed experiment
- [[Qwen2.5-0.5B-Instruct]] — verifier model
- [[Semantic Entropy]] — uncertainty metric used
- [[Conformal Risk Control]] — gating mechanism
- [[CrossEncoder]] — purpose-built reranker that works
- [[Selective Reranking]] — the approach that works
