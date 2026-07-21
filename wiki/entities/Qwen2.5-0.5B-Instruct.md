---
title: "Qwen2.5-0.5B-Instruct"
type: entity
tags: [llm, small-model, qwen, router, verifier]
sources: [phase1-phase2-report, seg-research-paper, phase3-progress-2026-06-24]
last_updated: 2026-06-29
---

## Summary
Qwen2.5-0.5B-Instruct is the small instruction-tuned LLM used in SEG for two roles: (1) as the query router backbone (QLoRA fine-tuned for route prediction), and (2) as the claim verifier for downstream-utility reranking (reading label log-probabilities for SUPPORT/REFUTE/NEI).

## Role 1: Query Router
- Fine-tuned with [[QLoRA]] on train split oracle labels
- Uses label log-probability scoring over bm25/dense/hybrid
- Accuracy: 0.4300 (raw), 0.3133 (calibrated held-out)
- Tends to overpredict Dense due to train/test distribution shift
- Calibrated with class bias, temperature, and margin threshold

## Role 2: Claim Verifier (Direction 3)
- Reads SUPPORT/REFUTE/NEI label log-probabilities in single forward pass
- Overconfident: non-relevant mean confidence 0.924
- Weak signal separation: AUC~0.60-0.64
- Inference: 37ms/pair GPU, 870ms/pair CPU
- Result: negative — cannot substitute for [[CrossEncoder]]

## Related To
- [[QLoRA]] — fine-tuning method
- [[Query Routing]] — router application
- [[Downstream-Utility Reranking]] — verifier application
- [[Semantic Entropy]] — uncertainty metric
