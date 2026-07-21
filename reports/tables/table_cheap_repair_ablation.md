# Cheap Post-Retrieval Repair Ablation

Date: 2026-07-20

This table summarizes the cheap post-retrieval repair path. The method keeps BGE-small-final as the default specialist and tests BM25-only repair before any extra dense/generalist, Cross-Encoder, or LLM stage.

## Dev Multi-Seed Summary

Source: `runs/fusion/dev_cheap_repair_multiseed_summary.csv`

| Stage | Method | Mean nDCG@10 | Mean Delta vs BGE-small | Delta Range | Mean Recall@10 | Mean Recall@100 | Mean MRR@10 | Mean Switch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | 0.0738 | -0.8385 | [-0.8490, -0.8238] | 0.1457 | 0.3893 | 0.0559 | N/A |
| S1 | BGE-small-final specialist | 0.9123 | 0.0000 | [0.0000, 0.0000] | 0.9926 | 1.0000 | 0.8897 | N/A |
| S2 | Confidence diagnostics only | 0.9123 | 0.0000 | [0.0000, 0.0000] | 0.9926 | 1.0000 | 0.8897 | 0.0000 |
| S3 | BM25 rerank inside specialist top-N | 0.9124 | +0.0001 | [-0.0045, +0.0051] | 0.9926 | 1.0000 | 0.8881 | 0.0000 |
| S4 | Always BM25 candidate injection | 0.9047 | -0.0076 | [-0.0378, +0.0000] | 0.9926 | 1.0000 | 0.8790 | 1.0000 |
| S5 | Conditional BM25 injection | 0.9119 | -0.0004 | [-0.0033, +0.0026] | 0.9926 | 1.0000 | 0.8879 | 0.1284 |
| S6 | Conditional lexical BM25 rescue | 0.9114 | -0.0009 | [-0.0046, +0.0000] | 0.9926 | 1.0000 | 0.8885 | 0.0444 |
| S7 | Bounded BM25 promotion | 0.9121 | -0.0002 | [-0.0009, +0.0000] | 0.9926 | 1.0000 | 0.8895 | 0.0222 |
| S8 | Rank-local BM25 promotion | 0.9116 | -0.0007 | [-0.0017, +0.0000] | 0.9926 | 1.0000 | 0.8891 | 0.0296 |
| S9 | Learned cheap gate no-op fallback | 0.9123 | +0.0000 | [+0.0000, +0.0000] | 0.9926 | 1.0000 | 0.8897 | 0.0000 |

## Test Diagnostic Multi-Seed Summary

Source: `runs/fusion/test_cheap_repair_multiseed_summary.csv`

| Stage | Method | Mean nDCG@10 | Mean Delta vs BGE-small | Delta Range | Mean Recall@10 | Mean Recall@100 | Mean MRR@10 | Mean Switch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | BM25 baseline | 0.6656 | -0.1544 | [-0.1923, -0.1202] | 0.7814 | 0.8761 | 0.6343 | N/A |
| S1 | BGE-small-final specialist | 0.8200 | 0.0000 | [0.0000, 0.0000] | 0.9419 | 0.9807 | 0.7868 | N/A |
| S2 | Confidence diagnostics only | 0.8200 | 0.0000 | [0.0000, 0.0000] | 0.9419 | 0.9807 | 0.7868 | 0.0000 |
| S3 | BM25 rerank inside specialist top-N | 0.8183 | -0.0017 | [-0.0216, +0.0071] | 0.9459 | 0.9807 | 0.7825 | 0.0000 |
| S4 | Always BM25 candidate injection | 0.8116 | -0.0084 | [-0.0341, +0.0028] | 0.9426 | 0.9833 | 0.7751 | 1.0000 |
| S5 | Conditional BM25 injection | 0.8167 | -0.0033 | [-0.0262, +0.0095] | 0.9409 | 0.9807 | 0.7830 | 0.1653 |
| S6 | Conditional lexical BM25 rescue | 0.8184 | -0.0016 | [-0.0075, +0.0012] | 0.9419 | 0.9807 | 0.7844 | 0.1040 |
| S7 | Bounded BM25 promotion | 0.8193 | -0.0007 | [-0.0009, +0.0000] | 0.9419 | 0.9807 | 0.7860 | 0.0213 |
| S8 | Rank-local BM25 promotion | 0.8205 | +0.0005 | [-0.0013, +0.0033] | 0.9432 | 0.9807 | 0.7874 | 0.1200 |
| S9 | Learned cheap gate + rank-local promotion | 0.8248 | +0.0048 | [-0.0017, +0.0083] | 0.9472 | 0.9807 | 0.7917 | 0.5720 |

## Interpretation

Cheap repair is not yet a robust performance contribution.

The strongest current evidence is diagnostic:

- BM25 can rescue a subset of specialist failures on test (`32/300` strict rescue queries).
- Oracle over BM25 + BGE-small-final reaches `0.8624`, or `+0.0436` over BGE-small-final.
- Rescueable queries have lower specialist top score and gap.

The hand-written rule family does not exploit that headroom reliably. S8 rank-local promotion is safer than S3 but still too conservative. S9 is the first cheap method with a meaningful diagnostic mean gain, but it cannot be claimed yet because clean trainfit->dev validation is missing.

## Failure Contribution Summary

Sources:

- `runs/fusion/dev_cheap_repair_failure_multiseed_summary.csv`
- `runs/fusion/test_cheap_repair_failure_multiseed_summary.csv`
- `runs/fusion/dev_cheap_repair_failure_case_profiles.csv`
- `runs/fusion/test_cheap_repair_failure_case_profiles.csv`

| Split | Stage | Changed@10 Mean | Helped Mean | Hurt Mean | Missed BM25 Rescue Mean | Mean Delta |
|---|---|---:|---:|---:|---:|---:|
| dev | S3 | 0.627 | 3.6 | 3.8 | 0.0 | +0.0001 |
| dev | S5 | 0.094 | 2.2 | 3.0 | 0.0 | -0.0004 |
| dev | S7 | 0.022 | 0.0 | 0.2 | 0.0 | -0.0002 |
| dev | S8 | 0.027 | 0.0 | 0.8 | 0.0 | -0.0007 |
| dev | S9 | 0.000 | 0.0 | 0.0 | 0.0 | +0.0000 |
| test | S3 | 0.955 | 14.0 | 12.2 | 0.4 | -0.0017 |
| test | S5 | 0.165 | 4.8 | 6.6 | 12.0 | -0.0033 |
| test | S7 | 0.019 | 0.0 | 0.8 | 16.0 | -0.0007 |
| test | S8 | 0.104 | 1.0 | 2.4 | 14.6 | +0.0005 |
| test | S9 | 0.551 | 9.2 | 13.4 | 4.2 | +0.0048 |

Interpretation:

- S3 has enough positive cases to be interesting, but its top-10 churn is too high.
- S7 is the safest current rule, but it is under-triggered and leaves most BM25 rescue headroom unused.
- S8 confirms that rank-local promotion alone is still under-triggered.
- S9 confirms that a learned cheap gate can reduce missed rescue cases, but the next proof must be trainfit->dev rather than test-derived calibration.
