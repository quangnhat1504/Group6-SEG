# Cheap Post-Retrieval Repair Ablation

Date: 2026-07-20

## Scope

This phase tests the cheap path before any expensive generalist, Cross-Encoder, or LLM rescue.

Rules:

- No query preprocessing.
- No Cross-Encoder reranking.
- No LLM reranking.
- BGE-small-final remains the default specialist.
- BM25 is used only as a cheap post-retrieval repair signal.
- Results selected from a split derived from `test` are diagnostic only, not final claims.

## Implemented Artifact

- Script: `scripts/run_cheap_repair_ablation.py`
- Clean validation script: `scripts/run_cheap_repair_clean_validation.py`
- Failure analyzer: `scripts/analyze_cheap_repair_failures.py`
- Aggregators: `scripts/summarize_cheap_repair_ablation.py`, `scripts/summarize_cheap_repair_failures.py`, `scripts/summarize_cheap_repair_case_profiles.py`
- Dev outputs: `runs/fusion/cheap_repair_dev_seed*/dev_cheap_repair_ablation.csv`
- Test diagnostic outputs: `runs/fusion/cheap_repair_seed*/test_cheap_repair_ablation.csv`
- Multi-seed summaries:
  - `runs/fusion/dev_cheap_repair_multiseed_summary.csv`
  - `runs/fusion/test_cheap_repair_multiseed_summary.csv`
- Failure summaries:
  - `runs/fusion/dev_cheap_repair_failure_multiseed_summary.csv`
  - `runs/fusion/test_cheap_repair_failure_multiseed_summary.csv`
- Case profiles:
  - `runs/fusion/dev_cheap_repair_failure_case_profiles.csv`
  - `runs/fusion/test_cheap_repair_failure_case_profiles.csv`
- Clean validation:
  - `runs/scifact/trainfit_dense_bge_small_scifact_rrf.csv`
  - `runs/fusion/cheap_repair_clean_validation/trainfit_to_dev_cheap_repair_clean_validation.csv`
  - `reports/tables/table_cheap_repair_clean_validation.md`

## Stage Design

| Stage | Method | Purpose |
|---|---|---|
| S0 | BM25 baseline | cheap lexical reference |
| S1 | BGE-small-final specialist | primary baseline to beat |
| S2 | Confidence diagnostics only | no ranking change; measure failure signals |
| S3 | BM25 rerank inside specialist top-N | cheap reorder without candidate injection |
| S4 | Always BM25 candidate injection | naive fusion baseline |
| S5 | Conditional BM25 injection | repair only when specialist confidence is low |
| S6 | Conditional lexical BM25 rescue | repair when specialist low-confidence and BM25 is sharp |
| S7 | Bounded BM25 promotion | protect specialist top ranks and promote at most 1-2 BM25 docs |
| S8 | Rank-local BM25 promotion | promote BM25-supported docs only into the shallow specialist tail |
| S9 | Learned cheap gate + rank-local promotion | logistic gate over cheap specialist/BM25 agreement features |

## Dev Multi-Seed Result

Source: `runs/fusion/dev_cheap_repair_multiseed_summary.csv`

Protocol: split SciFact `dev` into 50% calibration and 50% evaluation for seeds `1, 7, 13, 21, 42`.

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

### Dev Interpretation

- The specialist dominates dev: BM25 has no strict rescue headroom against BGE-small-final in this split.
- Cheap repair is not yet a reliable dev contribution.
- S3 is the only stage with a slightly positive mean delta, but the effect is effectively flat and not enough for a claim.
- S7 is safer than S4/S5/S6 because it limits damage, but it still does not beat the specialist.
- S8 is still slightly negative on dev.
- S9 becomes no-op on dev because the dev calibration subsets have no BM25 rescue-positive labels for the learned gate.
- Always-on BM25 injection hurts; naive fusion should not be the main direction.

## Test Diagnostic Multi-Seed Result

Source: `runs/fusion/test_cheap_repair_multiseed_summary.csv`

Protocol: split SciFact `test` into 50% calibration and 50% evaluation for seeds `1, 7, 13, 21, 42`. This is diagnostic only.

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

### Test Diagnostic Interpretation

- There is real BM25 rescue headroom on full test: oracle over BM25 + BGE-small-final is `0.8624`, or `+0.0436` over BGE-small-final.
- BM25 strictly beats the specialist on `32/300` queries.
- Rescueable queries have lower specialist confidence:
  - BGE-small top score: `0.6163` on rescueable vs `0.6759` on non-rescueable.
  - BGE-small gap: `0.0539` on rescueable vs `0.1130` on non-rescueable.
- However, the current cheap rules do not capture that headroom reliably across held-out seeds.
- S7 confirms that bounded promotion reduces downside but still fails to produce positive mean gain.
- S8 gives a tiny positive diagnostic test gain (`+0.0005`) but is not stable enough.
- S9 gives the strongest diagnostic test gain so far (`+0.0048` mean), but this is selected inside test-derived splits and cannot be a final result.

## False-Positive / False-Negative Analysis

Sources:

- `reports/tables/table_cheap_repair_failure_aggregate_dev.md`
- `reports/tables/table_cheap_repair_failure_aggregate_test.md`
- `reports/tables/table_cheap_repair_case_profiles_dev.md`
- `reports/tables/table_cheap_repair_case_profiles_test.md`

### Failure Aggregate

Dev:

| Stage | Changed@10 Mean | Helped Mean | Hurt Mean | Missed BM25 Rescue Mean | Mean Delta |
|---|---:|---:|---:|---:|---:|
| S3 | 0.627 | 3.6 | 3.8 | 0.0 | +0.0001 |
| S4 | 0.170 | 0.4 | 2.6 | 0.0 | -0.0076 |
| S5 | 0.094 | 2.2 | 3.0 | 0.0 | -0.0004 |
| S6 | 0.032 | 0.6 | 0.8 | 0.0 | -0.0009 |
| S7 | 0.022 | 0.0 | 0.2 | 0.0 | -0.0002 |
| S8 | 0.027 | 0.0 | 0.8 | 0.0 | -0.0007 |
| S9 | 0.000 | 0.0 | 0.0 | 0.0 | +0.0000 |

Test diagnostic:

| Stage | Changed@10 Mean | Helped Mean | Hurt Mean | Missed BM25 Rescue Mean | Mean Delta |
|---|---:|---:|---:|---:|---:|
| S3 | 0.955 | 14.0 | 12.2 | 0.4 | -0.0017 |
| S4 | 0.784 | 10.2 | 12.8 | 3.0 | -0.0084 |
| S5 | 0.165 | 4.8 | 6.6 | 12.0 | -0.0033 |
| S6 | 0.104 | 1.8 | 2.2 | 15.2 | -0.0016 |
| S7 | 0.019 | 0.0 | 0.8 | 16.0 | -0.0007 |
| S8 | 0.104 | 1.0 | 2.4 | 14.6 | +0.0005 |
| S9 | 0.551 | 9.2 | 13.4 | 4.2 | +0.0048 |

### What This Explains

- S3 is not a safe repair rule. It changes almost all test top-10 rankings (`0.955` changed@10), helps many queries, but causes enough false-positive hurt to erase the gain.
- S7 is safe but too conservative. It changes only `0.019` of test top-10 rankings, almost eliminates harm, but misses about `16` BM25 rescue opportunities per held-out seed.
- S5/S6 sit between those extremes. They reduce churn but still have more hurt than helped and miss most rescue opportunities.
- S8 remains too conservative: it helps only `1.0` query/seed on test and still misses `14.6` BM25 rescues.
- S9 is the first cheap method that unlocks a meaningful part of test diagnostic headroom, reducing missed BM25 rescues to `4.2` query/seed. The cost is a high switch rate (`0.572`) and many false-positive hurts (`13.4` query/seed), so it needs stricter clean validation.
- Dev has no BM25 rescue headroom under current artifacts, so any cheap rule must avoid harming dev while being selected using dev-only evidence. This is the main protocol tension.

### Case Profile Signals

From the test diagnostic case profiles:

- S3 helpful cases: specialist first relevant rank mean `5.07`, BM25 first relevant rank mean `3.34`, specialist gap `0.0552`.
- S3 harmful cases: specialist first relevant rank mean `3.23`, BM25 first relevant rank mean `9.05`, specialist gap `0.0250`.
- S7 missed rescue cases: specialist first relevant rank mean `4.93`, BM25 first relevant rank mean `1.71`, specialist gap `0.0490`.
- S7 true negatives: specialist first relevant rank mean `1.93`, BM25 first relevant rank mean `3.94`, specialist gap `0.1173`.
- S9 helpful cases: mean delta `+0.2321`, specialist first relevant rank mean `5.35`, BM25 first relevant rank mean `1.72`.
- S9 harmful cases: mean delta `-0.1057`, specialist first relevant rank mean `3.79`, BM25 first relevant rank mean `7.97`.

Interpretation: a cheap rescue signal exists, and a learned gate can find more of it than hand-written thresholds. The unresolved issue is clean calibration: the current dev artifact has no BM25 rescue-positive labels, so S9 cannot yet be selected by a valid trainfit/dev protocol.

## Current Leader Decision

Cheap post-retrieval repair has moved from pure diagnostic to a plausible method candidate, but it is still not a final performance contribution.

Do not claim:

> BM25 repair improves BGE-small-final.

Claim only:

> The failure region exists, BM25 can rescue a subset of specialist failures, and a learned cheap gate can unlock part of that headroom on diagnostic test-derived splits. Clean trainfit/dev validation is still missing.

## Clean Trainfit -> Dev Validation

Source: `reports/tables/table_cheap_repair_clean_validation.md`

Protocol: train/tune S8/S9 on SciFact `trainfit`, evaluate on held-out `dev`.

| Stage | Method | Selected On | Switch | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S1 | BGE-small-final specialist | fixed | N/A | 0.9052 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |
| S8 | Rank-local BM25 promotion | trainfit | 0.0124 | 0.9052 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |
| S9 | Learned cheap gate + rank-local promotion | trainfit | 0.0124 | 0.9057 | +0.0005 | 0.9938 | 1.0000 | 0.8799 |

Clean validation details:

- Generated `runs/scifact/trainfit_dense_bge_small_scifact_rrf.csv` from `runs/finetuned/bge-small-final`.
- Trainfit specialist nDCG@10: `0.9332`.
- Trainfit BM25 rescue-positive queries: `2`.
- Dev BM25 rescue-positive queries: `0`.
- Dev S9 changed `1` query, helped `1`, hurt `0`, and missed `0` BM25 rescues.

Decision: S9 does not meet the `+0.005` dev success criterion. The cheap BM25 repair path is now exhausted as the main performance path under the clean protocol.

## Next Ablation Plan

1. Move to Specialist-Generalist Adaptive Fusion / static ensemble validation.
2. Keep S9 as a diagnostic baseline and possible component, but do not freeze it as the main method.
3. Keep expensive fallback separate:
   - BGE-base/vstash can be compared later as an expensive rescue ablation, not part of the cheap path.

## Novelty Status

Potential novelty is not in BM25 reranking itself. The possible contribution is:

> Specialist-first retrieval repair: preserve a high-performing fine-tuned specialist by default, identify its low-confidence failure region, and apply bounded cheap repair only when the specialist is likely to fail.

After S0-S9 plus clean trainfit->dev validation, this is a strong diagnostic section but not the main performance contribution. The novelty should move to adaptive specialist-generalist selection/fusion unless a later cheap component can be validated on a split with real rescue headroom.

## Validation

- `python -m py_compile scripts/analyze_cheap_repair_failures.py scripts/summarize_cheap_repair_failures.py scripts/summarize_cheap_repair_case_profiles.py scripts/summarize_cheap_repair_ablation.py scripts/run_cheap_repair_ablation.py` passed.
- `python -m py_compile scripts/run_cheap_repair_clean_validation.py` passed.
- `python scripts/run_finetuned_dense_retrieval.py --config configs/scifact.yaml --query-split trainfit --documents-split train --model-path runs/finetuned/bge-small-final --output-run runs/scifact/trainfit_dense_bge_small_scifact_rrf.csv --output-metrics runs/scifact/trainfit_dense_bge_small_scifact_rrf_metrics.json` passed.
- `python scripts/run_cheap_repair_clean_validation.py --config configs/scifact.yaml --train-split trainfit --eval-split dev --output-dir runs/fusion/cheap_repair_clean_validation` passed.
- `python -m pytest tests/test_property_rrf.py -q` passed.
- Full `python -m pytest tests -q` currently fails during collection on legacy/unrelated tests:
  - missing `run_significance_conformal`;
  - `train_qpp_regressor.FEATURE_COLUMNS` was renamed to `FEATURE_COLS`;
  - `generate_comparison_table` is not exported from `train_qpp_regressor`.
