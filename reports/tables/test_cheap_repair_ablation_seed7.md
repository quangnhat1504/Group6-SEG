# Cheap Post-Retrieval Repair Ablation

- Split: `test`
- Calibration fraction: `0.5`
- Seed: `7`
- Primary metric: `nDCG@10`
- Note: this is a lightweight post-retrieval path; no query preprocessing, no cross-encoder, no LLM reranker.

## Contribution Table

| Stage | Method | Selected On | Cost | Switch Rate | nDCG@10 | Delta vs Prev | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | fixed | 1.00 | N/A | 0.6660 | N/A | -0.1679 | 0.7833 | 0.8750 | 0.6322 |
| S1 | BGE-small-final specialist | fixed | 3.00 | N/A | 0.8339 | 0.1679 | 0.0000 | 0.9633 | 0.9933 | 0.7969 |
| S2 | Confidence diagnostics only | fixed | 3.00 | 0.0000 | 0.8339 | 0.0000 | 0.0000 | 0.9633 | 0.9933 | 0.7969 |
| S3 | BM25 rerank inside specialist top-N | calibration | 3.10 | 0.0000 | 0.8305 | -0.0034 | -0.0034 | 0.9567 | 0.9933 | 0.7941 |
| S4 | Always BM25 candidate injection | calibration | 3.20 | 1.0000 | 0.8230 | -0.0075 | -0.0109 | 0.9500 | 0.9933 | 0.7851 |
| S5 | Conditional BM25 injection | calibration | 3.20 | 0.0533 | 0.8320 | 0.0091 | -0.0018 | 0.9500 | 0.9933 | 0.7987 |
| S6 | Conditional lexical BM25 rescue | calibration | 3.20 | 0.1133 | 0.8263 | -0.0057 | -0.0075 | 0.9500 | 0.9933 | 0.7909 |
| S7 | Bounded BM25 promotion | calibration | 3.20 | 0.0133 | 0.8330 | 0.0067 | -0.0009 | 0.9633 | 0.9933 | 0.7958 |
| S8 | Rank-local BM25 promotion | calibration | 3.20 | 0.1533 | 0.8326 | -0.0004 | -0.0013 | 0.9633 | 0.9933 | 0.7955 |
| S9 | Learned cheap gate + rank-local promotion | calibration | 3.30 | 0.5400 | 0.8322 | -0.0004 | -0.0017 | 0.9633 | 0.9933 | 0.7939 |

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
