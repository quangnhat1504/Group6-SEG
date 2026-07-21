---
title: "Semantic Entropy"
type: concept
tags: [llm-uncertainty, entropy, verification, reranking]
sources: [seg-research-paper, phase3-progress-2026-06-24]
last_updated: 2026-06-29
---

## Summary
Semantic entropy (Farquhar et al., Nature 2024) measures LLM uncertainty by clustering sampled outputs into semantic equivalence classes. In SEG's [[Downstream-Utility Reranking]], it was used as a reranking signal: documents that make the LLM more decisive (lower entropy) about claim verification get higher scores.

## Application in SEG
- Used with [[Qwen2.5-0.5B-Instruct]] for claim verification
- Computed from label distribution over SUPPORT/REFUTE/NEI
- Negative entropy used as "decisiveness" score
- InfoGain variant: change in decisiveness vs claim-only baseline
- Result: weak signal separation (AUC~0.60-0.64)

## Limitations
- 0.5B model overconfident: non-relevant mean confidence 0.924
- Discrete sampling collapses to constant NEI on small model
- Log-probability method preferred over sampling

## Related To
- [[Downstream-Utility Reranking]] — application
- [[Qwen2.5-0.5B-Instruct]] — verifier model
- [[Conformal Risk Control]] — gating mechanism
