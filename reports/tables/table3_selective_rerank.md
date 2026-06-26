| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Rerank Coverage | Est. ms/query |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6523 | 0.7757 | 0.8731 | 0.6184 | 0.00 | 0.0 |
| Dense / SciNCL | 0.5640 | 0.7233 | 0.9082 | 0.5224 | 0.00 | 0.0 |
| Hybrid RRF | 0.6583 | 0.8146 | 0.9560 | 0.6157 | 0.00 | 0.0 |
| Selective rerank (score-gap, headline) | 0.6924 | 0.8186 | 0.9560 | 0.6611 | 0.66 | 18.1 |
| Selective rerank (config default OR-rule) | 0.6977 | 0.8286 | 0.9560 | 0.6649 | 0.82 | 22.5 |
| Always-Rerank | 0.6939 | 0.8286 | 0.9560 | 0.6604 | 1.00 | 27.3 |
