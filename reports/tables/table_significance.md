Paired bootstrap significance tests on SciFact test (10000 resamples, seed=13). CI = 95% percentile interval of the mean paired difference; p = two-sided bootstrap p-value. Bold = CI excludes 0.

| Comparison | n | Metric | System | Baseline | Mean diff | 95% CI | p |
|---|---:|---|---:|---:|---:|---|---:|
| Hybrid RRF vs BM25 | 300 | ndcg@10 | 0.6583 | 0.6523 | +0.0060 | [-0.0223, +0.0342] | 0.6648 |
| Hybrid RRF vs BM25 | 300 | mrr@10 | 0.6157 | 0.6184 | -0.0027 | [-0.0350, +0.0296] | 0.8746 |
| Always-Rerank vs Hybrid RRF | 300 | ndcg@10 | 0.6939 | 0.6583 | **+0.0356** | [+0.0076, +0.0628] | 0.0120 |
| Always-Rerank vs Hybrid RRF | 300 | mrr@10 | 0.6604 | 0.6157 | **+0.0447** | [+0.0108, +0.0803] | 0.0088 |
| Oracle Router vs Hybrid RRF | 300 | ndcg@10 | 0.7617 | 0.6583 | **+0.1034** | [+0.0798, +0.1278] | 0.0000 |
| Oracle Router vs Hybrid RRF | 300 | mrr@10 | 0.7337 | 0.6157 | **+0.1180** | [+0.0913, +0.1471] | 0.0000 |
| Calibrated LLM (held-out) vs BM25 | 150 | ndcg@10 | 0.6674 | 0.6489 | +0.0186 | [-0.0110, +0.0476] | 0.2130 |
| Calibrated LLM (held-out) vs BM25 | 150 | mrr@10 | 0.6259 | 0.6184 | +0.0075 | [-0.0252, +0.0402] | 0.6396 |
