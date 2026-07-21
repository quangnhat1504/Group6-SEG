# Cheap Post-Retrieval Repair Ablation

- Split: `test`
- Calibration fraction: `0.5`
- Seed: `1`
- Primary metric: `nDCG@10`
- Note: this is a lightweight post-retrieval path; no query preprocessing, no cross-encoder, no LLM reranker.

## Contribution Table

| Stage | Method | Selected On | Cost | Switch Rate | nDCG@10 | Delta vs Prev | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | fixed | 1.00 | N/A | 0.7157 | N/A | -0.1202 | 0.8367 | 0.9061 | 0.6832 |
| S1 | BGE-small-final specialist | fixed | 3.00 | N/A | 0.8360 | 0.1202 | 0.0000 | 0.9711 | 0.9800 | 0.7973 |
| S2 | Confidence diagnostics only | fixed | 3.00 | 0.0000 | 0.8360 | 0.0000 | 0.0000 | 0.9711 | 0.9800 | 0.7973 |
| S3 | BM25 rerank inside specialist top-N | calibration | 3.10 | 0.0000 | 0.8411 | 0.0051 | 0.0051 | 0.9711 | 0.9800 | 0.8035 |
| S4 | Always BM25 candidate injection | calibration | 3.20 | 1.0000 | 0.8335 | -0.0075 | -0.0024 | 0.9644 | 0.9800 | 0.7952 |
| S5 | Conditional BM25 injection | calibration | 3.20 | 0.0600 | 0.8455 | 0.0119 | 0.0095 | 0.9644 | 0.9800 | 0.8119 |
| S6 | Conditional lexical BM25 rescue | calibration | 3.20 | 0.1600 | 0.8351 | -0.0104 | -0.0009 | 0.9711 | 0.9800 | 0.7959 |
| S7 | Bounded BM25 promotion | calibration | 3.20 | 0.0333 | 0.8351 | 0.0000 | -0.0009 | 0.9711 | 0.9800 | 0.7962 |
| S8 | Rank-local BM25 promotion | calibration | 3.20 | 0.1800 | 0.8368 | 0.0017 | 0.0008 | 0.9711 | 0.9800 | 0.7990 |
| S9 | Learned cheap gate + rank-local promotion | calibration | 3.30 | 0.6200 | 0.8380 | 0.0012 | 0.0020 | 0.9711 | 0.9800 | 0.8009 |

## Diagnostics

- Specialist failure-rescue query count: `32/300`
- Oracle over BM25 + BGE-small-final: `0.8624`
- Oracle headroom vs BGE-small-final: `+0.0436`
- Average BGE-small top score on rescueable queries: `0.6163`
- Average BGE-small top score on non-rescueable queries: `0.6759`
- Average BGE-small gap on rescueable queries: `0.0539`
- Average BGE-small gap on non-rescueable queries: `0.1130`

## Current Interpretation

- The cheap path has real headroom because BM25 rescues a subset of specialist failures.
- Naive always-on fusion is a necessary baseline, but the target contribution is conditional repair: preserve the specialist when confident and intervene only on low-confidence or lexical-sharp queries.
- Any result selected on a split derived from `test` is diagnostic only. The final claim still requires trainfit/dev selection, frozen config, one test run, and duplicate-filtered sensitivity.

## Next Phase

1. Generate missing `dev`/`trainfit` run files for BGE-small-final if needed.
2. Re-run this script on `dev` or a true trainfit/dev protocol.
3. Freeze the best cheap repair rule before touching SciFact test again.
4. Only after cheap BM25 repair is exhausted, compare optional dense/generalist fallback as a separate expensive ablation.
