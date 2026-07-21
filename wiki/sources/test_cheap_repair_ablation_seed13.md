# Cheap Post-Retrieval Repair Ablation

- Split: `test`
- Calibration fraction: `0.5`
- Seed: `13`
- Primary metric: `nDCG@10`
- Note: this is a lightweight post-retrieval path; no query preprocessing, no cross-encoder, no LLM reranker.

## Contribution Table

| Stage | Method | Selected On | Cost | Switch Rate | nDCG@10 | Delta vs Prev | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | fixed | 1.00 | N/A | 0.6489 | N/A | -0.1525 | 0.7656 | 0.8691 | 0.6184 |
| S1 | BGE-small-final specialist | fixed | 3.00 | N/A | 0.8013 | 0.1525 | 0.0000 | 0.9344 | 0.9867 | 0.7632 |
| S2 | Confidence diagnostics only | fixed | 3.00 | 0.0000 | 0.8013 | 0.0000 | 0.0000 | 0.9344 | 0.9867 | 0.7632 |
| S3 | BM25 rerank inside specialist top-N | calibration | 3.10 | 0.0000 | 0.8085 | 0.0071 | 0.0071 | 0.9478 | 0.9867 | 0.7677 |
| S4 | Always BM25 candidate injection | calibration | 3.20 | 1.0000 | 0.8038 | -0.0047 | 0.0024 | 0.9411 | 0.9867 | 0.7645 |
| S5 | Conditional BM25 injection | calibration | 3.20 | 0.2933 | 0.8000 | -0.0037 | -0.0013 | 0.9478 | 0.9867 | 0.7564 |
| S6 | Conditional lexical BM25 rescue | calibration | 3.20 | 0.0933 | 0.8007 | 0.0007 | -0.0006 | 0.9344 | 0.9867 | 0.7615 |
| S7 | Bounded BM25 promotion | calibration | 3.20 | 0.0067 | 0.8013 | 0.0006 | 0.0000 | 0.9344 | 0.9867 | 0.7632 |
| S8 | Rank-local BM25 promotion | calibration | 3.20 | 0.0267 | 0.8008 | -0.0005 | -0.0005 | 0.9344 | 0.9867 | 0.7632 |
| S9 | Learned cheap gate + rank-local promotion | calibration | 3.30 | 0.5400 | 0.8095 | 0.0088 | 0.0082 | 0.9411 | 0.9867 | 0.7726 |

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
