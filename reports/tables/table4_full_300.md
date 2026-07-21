# Table 4 — Full 300 SciFact Test Queries

All systems evaluated on the **same 300 test queries**. CRC Selective calibrated on train set (809 queries), α=0.02.

| Model | nDCG@10 | MAP@10 | Recall@10 | MRR@10 | Coverage |
|-------|---------|--------|-----------|--------|----------|
| BM25 | 0.6523 | 0.6071 | 0.7757 | 0.6184 | 0.00 |
| BM25_RM3 | 0.0099 | 0.0059 | 0.0225 | 0.0061 | 0.00 |
| Dense (SciNCL) | 0.5640 | 0.5077 | 0.7233 | 0.5224 | 0.00 |
| Dense (SPECTER) | 0.3523 | 0.3019 | 0.5004 | 0.3133 | 0.00 |
| Dense (SciBERT) | 0.1300 | 0.1023 | 0.2069 | 0.1152 | 0.00 |
| Hybrid (SciNCL + BM25) | 0.6583 | 0.6022 | 0.8146 | 0.6157 | 0.00 |
| Hybrid (SPECTER + BM25) | 0.3863 | 0.3274 | 0.5562 | 0.3421 | 0.00 |
| Hybrid (SciBERT + BM25) | 0.4248 | 0.3444 | 0.6707 | 0.3612 | 0.00 |
| Hybrid (SciNCL + BM25) + CE | 0.6939 | 0.6448 | 0.8286 | 0.6604 | 1.00 |
| Hybrid (SPECTER + BM25) + CE | 0.5662 | 0.5389 | 0.6321 | 0.5544 | 1.00 |
| Hybrid (SciBERT + BM25) + CE | 0.6838 | 0.6415 | 0.7941 | 0.6579 | 1.00 |
| CRC Selective (α=0.02) | 0.6740 | 0.6195 | 0.8263 | 0.6325 | 0.14 |

**Fair comparison:** Always-Rerank nDCG@10 = 0.6939, Selective nDCG@10 = 0.6740, shortfall = 0.0199 ≤ α=0.02 → guarantee holds ✓

Coverage = fraction of queries reranked (0.0 = no reranking, 1.0 = all reranked).
