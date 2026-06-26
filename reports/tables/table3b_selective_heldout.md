Threshold tuned on 150 calibration queries (seed=13), evaluated on 150 disjoint queries. Signal: score_gap, tuned threshold=0.002371.

| Method (eval subset) | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Rerank Coverage | Est. ms/query |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid RRF (no rerank) | 0.6860 | 0.8311 | 0.9787 | 0.6457 | 0.00 | 0.0 |
| Selective rerank (held-out tuned) | 0.7145 | 0.8224 | 0.9787 | 0.6892 | 0.74 | 20.2 |
| Always-Rerank | 0.7181 | 0.8358 | 0.9787 | 0.6907 | 1.00 | 27.3 |
