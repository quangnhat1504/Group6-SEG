# Rerank Depth Ablation: Top-20 vs Top-50

Paired bootstrap significance test (10000 resamples, seed=42) on nDCG@10.

| Metric | Top-20 | Top-50 | Δ | Significant |
|--------|-------:|-------:|--:|:-----------:|
| nDCG@10 | 0.6939 | 0.6916 | -0.0023 | No |
| Recall@10 | 0.8286 | 0.8189 | -0.0097 | — |
| MRR@10 | 0.6604 | 0.6597 | -0.0008 | — |
| Latency (ms/query) | 27.29 | 52.48 | — | — |

**Bootstrap CI (nDCG@10 diff):** [-0.0141, +0.0103], p = 0.6762

**Note:** Al-Joofi et al. (2025) uses top-100 depth. Direct comparison of absolute nDCG is not valid due to protocol differences (100 vs 300 queries, different base systems).
