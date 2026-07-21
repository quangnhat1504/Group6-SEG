---
title: "Downstream-Utility Reranking"
type: concept
tags: [llm-reranking, claim-verification, negative-result]
sources: [seg-research-paper, seg-research-directions, seg-research-deep-plan, phase3-progress-2026-06-24, validation-experiments]
last_updated: 2026-06-29
---

## Summary
Phase 4 Direction 3: rerank documents not by relevance but by how much they help a small LLM verify the scientific claim (downstream utility). Implemented with Qwen2.5-0.5B-Instruct reading log-probability distributions over SUPPORT/REFUTE/NEI.

## Results — NEGATIVE
- Signal separation: AUC~0.60-0.64 (barely above random)
- 0.5B verifier is overconfident (non-relevant mean confidence 0.924)
- Pure reorder: nDCG@10=0.351 (vs Hybrid base 0.6583)
- Best blend: +0.0006 (within noise)
- Conformal-gated selective: worse at every coverage
- Inference: 37ms/pair GPU, 870ms/pair CPU

## Signal Types
- Verification confidence: max·(1−P(NEI))
- Negative semantic entropy: distributional decisiveness
- InfoGain: change vs claim-only baseline

## Conclusion
Small generative verifier cannot substitute for purpose-built cross-encoder. Strengthens thesis that value is in cheap signals deciding when to invoke strong reranker, not making expensive stage reason about downstream task. Suggested follow-up: retry with Qwen2.5-1.5B/3B.

## Related To
- [[Qwen2.5-0.5B-Instruct]] — verifier model
- [[Semantic Entropy]] — uncertainty metric
- [[Selective Reranking]] — gating mechanism used
- [[CrossEncoder]] — purpose-built reranker that works
- [[Conformal Risk Control]] — gating for this experiment
