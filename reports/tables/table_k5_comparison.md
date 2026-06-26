# RRF k=5 vs k=60 Pipeline Comparison

Evaluation on SciFact test split (300 queries total, 150 eval subset). CRC: seed=13, alpha=0.02, signal=hybrid_max.

## Full-Set Metrics (all queries)

| Metric | k=60 | k=5 | Δ (k5 − k60) |
|--------|-----:|----:|-------------:|
| Hybrid nDCG@10 | 0.6583 | 0.6809 | +0.0226 |
| Hybrid Recall@100 | 0.9560 | 0.9560 | +0.0000 |
| Always-Rerank nDCG@10 | 0.6939 | 0.7014 | +0.0075 |
| Always-Rerank MRR@10 | 0.6604 | 0.6659 | +0.0055 |

## Eval-Subset Metrics (CRC evaluation split)

| Metric | k=60 | k=5 | Δ (k5 − k60) |
|--------|-----:|----:|-------------:|
| Hybrid nDCG@10 (eval) | 0.6860 | 0.7060 | +0.0201 |
| Always-Rerank nDCG@10 (eval) | 0.7181 | 0.7321 | +0.0140 |
| Conformal Selective nDCG@10 | 0.7280 | 0.7408 | +0.0128 |
| Rerank Coverage | 0.61 | 0.61 | +0.00 |

## QPP Signal Analysis

- Top QPP signal for gain-from-reranking on k=5: **hybrid_max** (|Kendall|=0.100)
- hybrid_max remains the top predictor on k=5 base (consistent with k=60)

## Key Findings

**CONFIRMATION**: Conformal selective reranking on k=5 (nDCG@10=0.7408) surpasses Always-Rerank on k=5 (nDCG@10=0.7321) by +0.0088. This confirms selective reranking can improve over always-reranking regardless of base retriever strength.

k=5 produces a stronger hybrid base than k=60 (+0.0226 nDCG@10), confirming the RRF k-dilution finding from Al-Joofi et al.
