Direction 3: downstream-utility reranking with Qwen2.5-0.5B-Instruct on SciFact (test, 300 queries, top-20 candidates). Negative result.

Signal separation of relevant vs non-relevant candidates (higher AUC = more useful):

| Signal | AUC | mean (relevant) | mean (non-relevant) |
|---|---:|---:|---:|
| confidence | 0.638 | 0.959 | 0.924 |
| neg_entropy | 0.633 | -0.096 | -0.155 |
| infogain_decision | 0.632 | 0.003 | -0.015 |
| infogain_entropy | 0.597 | 0.145 | 0.085 |

Reranking nDCG@10 (base Hybrid RRF = 0.6583); the LLM-utility signal never improves it:

| Variant | nDCG@10 | Δ vs base |
|---|---:|---:|
| Hybrid base (no rerank) | 0.6583 | — |
| Always-Utility, confidence (pure reorder) | 0.3509 | -0.3074 |
| Always-Utility, InfoGain-decision (pure reorder) | 0.3579 | -0.3004 |
| Best blend, InfoGain-decision (w=0.3) | 0.6589 | +0.0006 |
| Selective, InfoGain-decision @61% coverage | 0.5396 | -0.1187 |

For comparison, the cross-encoder Always-Rerank reaches nDCG@10 ≈ 0.728 (Section 6.9): a purpose-built relevance reranker succeeds where the generative verifier does not.
