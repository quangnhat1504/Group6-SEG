# Cheap Post-Retrieval Repair Ablation

- Split: `dev`
- Calibration fraction: `0.5`
- Seed: `13`
- Primary metric: `nDCG@10`
- Note: this is a lightweight post-retrieval path; no query preprocessing, no cross-encoder, no LLM reranker.

## Contribution Table

| Stage | Method | Selected On | Cost | Switch Rate | nDCG@10 | Delta vs Prev | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | fixed | 1.00 | N/A | 0.0886 | N/A | -0.8238 | 0.1698 | 0.4084 | 0.0688 |
| S1 | BGE-small-final specialist | fixed | 3.00 | N/A | 0.9124 | 0.8238 | 0.0000 | 0.9877 | 1.0000 | 0.8909 |
| S2 | Confidence diagnostics only | fixed | 3.00 | 0.0000 | 0.9124 | 0.0000 | 0.0000 | 0.9877 | 1.0000 | 0.8909 |
| S3 | BM25 rerank inside specialist top-N | calibration | 3.10 | 0.0000 | 0.9079 | -0.0045 | -0.0045 | 0.9877 | 1.0000 | 0.8817 |
| S4 | Always BM25 candidate injection | calibration | 3.20 | 1.0000 | 0.9124 | 0.0045 | 0.0000 | 0.9877 | 1.0000 | 0.8909 |
| S5 | Conditional BM25 injection | calibration | 3.20 | 0.1111 | 0.9141 | 0.0018 | 0.0018 | 0.9877 | 1.0000 | 0.8909 |
| S6 | Conditional lexical BM25 rescue | calibration | 3.20 | 0.0494 | 0.9078 | -0.0063 | -0.0046 | 0.9877 | 1.0000 | 0.8848 |
| S7 | Bounded BM25 promotion | calibration | 3.20 | 0.0123 | 0.9124 | 0.0046 | 0.0000 | 0.9877 | 1.0000 | 0.8909 |
| S8 | Rank-local BM25 promotion | calibration | 3.20 | 0.0247 | 0.9114 | -0.0010 | -0.0010 | 0.9877 | 1.0000 | 0.8909 |
| S9 | Learned cheap gate no-op fallback | calibration | 3.30 | 0.0000 | 0.9124 | 0.0010 | 0.0000 | 0.9877 | 1.0000 | 0.8909 |

## Diagnostics

- Specialist failure-rescue query count: `0/161`
- Oracle over BM25 + BGE-small-final: `0.9052`
- Oracle headroom vs BGE-small-final: `+0.0000`
- Average BGE-small top score on rescueable queries: `0.0000`
- Average BGE-small top score on non-rescueable queries: `0.6663`
- Average BGE-small gap on rescueable queries: `0.0000`
- Average BGE-small gap on non-rescueable queries: `0.1162`

## Current Interpretation

- The cheap path has real headroom because BM25 rescues a subset of specialist failures.
- Naive always-on fusion is a necessary baseline, but the target contribution is conditional repair: preserve the specialist when confident and intervene only on low-confidence or lexical-sharp queries.
- Any result selected on a split derived from `test` is diagnostic only. The final claim still requires trainfit/dev selection, frozen config, one test run, and duplicate-filtered sensitivity.

## Next Phase

1. Generate missing `dev`/`trainfit` run files for BGE-small-final if needed.
2. Re-run this script on `dev` or a true trainfit/dev protocol.
3. Freeze the best cheap repair rule before touching SciFact test again.
4. Only after cheap BM25 repair is exhausted, compare optional dense/generalist fallback as a separate expensive ablation.
