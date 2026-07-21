# Computational Cost Comparison

SciFact test: 300 queries, 5183 docs. Device: cuda:0. Warm-up: 10 queries.

## Measured (current setup)

| Pipeline | Setup Time (s) | Avg Latency (ms/query) | CPU (%) | GPU (%) | n_queries |
|----------|---------------|----------------------|---------|---------|-----------|
| BM25 (lexical) | 0.14 | 333.58 | 6.9 | 0.7 | 300 |
| Dense / SciNCL (cuda:0) | 29.69 | 6.11 | 5.8 | 35.1 | 300 |
| Hybrid RRF (BM25+Dense) | 29.83 | 354.77 | 6.8 | 3.4 | 300 |
| CE rerank only (top-20) | 8.18 | 24.86 | 6.8 | 70.0 | 300 |
| Always-Rerank (Hybrid+CE) | 38.01 | 379.62 | 6.8 | 36.7 | 300 |
| CRC Selective (α=0.02) | 38.01 | 358.14 | 1.2 | 70.1 | 300 |

**CRC Selective** reranks only 14% of queries, saving **5.7%** latency vs Always-Rerank.

## Projected with Optimized BM25 (Lucene/Elasticsearch)

If BM25 uses Lucene (~2.0ms) instead of Python rank_bm25 (~333.58ms):

| Pipeline | Avg Latency (ms/query) | Notes |
|----------|----------------------|-------|
| BM25 (Lucene/ES) | 2.0 | estimated |
| Dense / SciNCL (GPU) | 6.11 | measured |
| Hybrid RRF (Lucene+Dense+fuse) | 9.11 | projected |
| CE rerank (top-20) | 24.86 | measured |
| Always-Rerank (projected) | 33.97 | hybrid + CE all |
| CRC Selective α=0.02 (projected) | 12.51 | 14% CE, save 63.2% |

**CRC Selective saves 63.2% total latency** when BM25 is not the bottleneck.
CE dominates (24.9ms out of 34.0ms = 73% of pipeline),
so skipping 86% CE calls yields substantial wall-clock savings.
