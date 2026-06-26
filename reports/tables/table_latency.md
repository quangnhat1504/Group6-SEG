Measured per-query online latency on SciFact test (5183 docs, 300 queries, batch=1, after 10 warm-up queries). Offline one-time costs listed separately.

| Stage | ms/query (online) | Offline one-time |
|---|---:|---|
| BM25 (rank_bm25) | 11.25 | index build 0.3s |
| Dense / SciNCL | 16.52 | doc encode 54.0s (cuda:0) |
| Hybrid RRF (BM25+Dense+fuse) | 33.97 | - |
| Cross-encoder rerank (Hybrid top-20, GPU) | 27.3 | - (from Always-Rerank) |
