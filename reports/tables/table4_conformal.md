Conformal risk control at alpha = 0.02 (expected nDCG shortfall vs Always-Rerank), delta via CRC expectation bound. Threshold tuned on 150 calibration queries (seed=13), evaluated on 150 disjoint queries.

Reference: Hybrid (no rerank) nDCG@10 = 0.6860; Always-Rerank nDCG@10 = 0.7181.

| Trigger signal | Rerank Coverage | nDCG@10 | Realized risk | Guarantee (<= alpha) | Est. ms/query |
|---|---:|---:|---:|:--:|---:|
| QPP: hybrid max score (Direction 2) | 0.61 | 0.7280 | 0.0052 | yes | 16.6 |
| Phase-3 heuristic: score gap | 0.72 | 0.7120 | 0.0152 | yes | 19.6 |
