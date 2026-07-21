# Cheap Post-Retrieval Repair Ablation

- Split: `test`
- Calibration fraction: `0.5`
- Seed: `42`
- Primary metric: `nDCG@10`
- Note: this is a lightweight post-retrieval path; no query preprocessing, no cross-encoder, no LLM reranker.

## Contribution Table

| Stage | Method | Selected On | Cost | Switch Rate | nDCG@10 | Delta vs Prev | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | fixed | 1.00 | N/A | 0.6824 | N/A | -0.1390 | 0.8024 | 0.8787 | 0.6463 |
| S1 | BGE-small-final specialist | fixed | 3.00 | N/A | 0.8215 | 0.1390 | 0.0000 | 0.9187 | 0.9633 | 0.7991 |
| S2 | Confidence diagnostics only | fixed | 3.00 | 0.0000 | 0.8215 | 0.0000 | 0.0000 | 0.9187 | 0.9633 | 0.7991 |
| S3 | BM25 rerank inside specialist top-N | calibration | 3.10 | 0.0000 | 0.8257 | 0.0043 | 0.0043 | 0.9320 | 0.9633 | 0.7997 |
| S4 | Always BM25 candidate injection | calibration | 3.20 | 1.0000 | 0.8242 | -0.0015 | 0.0028 | 0.9253 | 0.9633 | 0.8007 |
| S5 | Conditional BM25 injection | calibration | 3.20 | 0.0667 | 0.8249 | 0.0007 | 0.0034 | 0.9187 | 0.9633 | 0.8037 |
| S6 | Conditional lexical BM25 rescue | calibration | 3.20 | 0.0200 | 0.8215 | -0.0034 | 0.0000 | 0.9187 | 0.9633 | 0.7991 |
| S7 | Bounded BM25 promotion | calibration | 3.20 | 0.0200 | 0.8206 | -0.0009 | -0.0009 | 0.9187 | 0.9633 | 0.7980 |
| S8 | Rank-local BM25 promotion | calibration | 3.20 | 0.1800 | 0.8248 | 0.0042 | 0.0033 | 0.9253 | 0.9633 | 0.8015 |
| S9 | Learned cheap gate + rank-local promotion | calibration | 3.30 | 0.5067 | 0.8298 | 0.0050 | 0.0083 | 0.9320 | 0.9633 | 0.8055 |

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
