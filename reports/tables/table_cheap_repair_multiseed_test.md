# Cheap Repair Multi-Seed Summary

Split: `test`

| Stage | Method | Runs | Mean nDCG@10 | Delta Mean | Delta Range | Recall@10 | Recall@100 | MRR@10 | Switch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | 5 | 0.6656 | -0.1544 | [-0.1923, -0.1202] | 0.7814 | 0.8761 | 0.6343 | N/A |
| S1 | BGE-small-final specialist | 5 | 0.8200 | +0.0000 | [+0.0000, +0.0000] | 0.9419 | 0.9807 | 0.7868 | N/A |
| S2 | Confidence diagnostics only | 5 | 0.8200 | +0.0000 | [+0.0000, +0.0000] | 0.9419 | 0.9807 | 0.7868 | 0.0000 |
| S3 | BM25 rerank inside specialist top-N | 5 | 0.8183 | -0.0017 | [-0.0216, +0.0071] | 0.9459 | 0.9807 | 0.7825 | 0.0000 |
| S4 | Always BM25 candidate injection | 5 | 0.8116 | -0.0084 | [-0.0341, +0.0028] | 0.9426 | 0.9833 | 0.7751 | 1.0000 |
| S5 | Conditional BM25 injection | 5 | 0.8167 | -0.0033 | [-0.0262, +0.0095] | 0.9409 | 0.9807 | 0.7830 | 0.1653 |
| S6 | Conditional lexical BM25 rescue | 5 | 0.8184 | -0.0016 | [-0.0075, +0.0012] | 0.9419 | 0.9807 | 0.7844 | 0.1040 |
| S7 | Bounded BM25 promotion | 5 | 0.8193 | -0.0007 | [-0.0009, +0.0000] | 0.9419 | 0.9807 | 0.7860 | 0.0213 |
| S8 | Rank-local BM25 promotion | 5 | 0.8205 | +0.0005 | [-0.0013, +0.0033] | 0.9432 | 0.9807 | 0.7874 | 0.1200 |
| S9 | Learned cheap gate + rank-local promotion | 5 | 0.8248 | +0.0048 | [-0.0017, +0.0083] | 0.9472 | 0.9807 | 0.7917 | 0.5720 |