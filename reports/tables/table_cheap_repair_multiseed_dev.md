# Cheap Repair Multi-Seed Summary

Split: `dev`

| Stage | Method | Runs | Mean nDCG@10 | Delta Mean | Delta Range | Recall@10 | Recall@100 | MRR@10 | Switch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | 5 | 0.0738 | -0.8385 | [-0.8490, -0.8238] | 0.1457 | 0.3893 | 0.0559 | N/A |
| S1 | BGE-small-final specialist | 5 | 0.9123 | +0.0000 | [+0.0000, +0.0000] | 0.9926 | 1.0000 | 0.8897 | N/A |
| S2 | Confidence diagnostics only | 5 | 0.9123 | +0.0000 | [+0.0000, +0.0000] | 0.9926 | 1.0000 | 0.8897 | 0.0000 |
| S3 | BM25 rerank inside specialist top-N | 5 | 0.9124 | +0.0001 | [-0.0045, +0.0051] | 0.9926 | 1.0000 | 0.8881 | 0.0000 |
| S4 | Always BM25 candidate injection | 5 | 0.9047 | -0.0076 | [-0.0378, +0.0000] | 0.9926 | 1.0000 | 0.8790 | 1.0000 |
| S5 | Conditional BM25 injection | 5 | 0.9119 | -0.0004 | [-0.0033, +0.0026] | 0.9926 | 1.0000 | 0.8879 | 0.1284 |
| S6 | Conditional lexical BM25 rescue | 5 | 0.9114 | -0.0009 | [-0.0046, +0.0000] | 0.9926 | 1.0000 | 0.8885 | 0.0444 |
| S7 | Bounded BM25 promotion | 5 | 0.9121 | -0.0002 | [-0.0009, +0.0000] | 0.9926 | 1.0000 | 0.8895 | 0.0222 |
| S8 | Rank-local BM25 promotion | 5 | 0.9116 | -0.0007 | [-0.0017, +0.0000] | 0.9926 | 1.0000 | 0.8891 | 0.0296 |
| S9 | Learned cheap gate no-op fallback | 5 | 0.9123 | +0.0000 | [+0.0000, +0.0000] | 0.9926 | 1.0000 | 0.8897 | 0.0000 |