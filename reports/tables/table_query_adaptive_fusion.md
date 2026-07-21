# Query-Adaptive Specialist-Generalist Fusion

Date: 2026-07-20

Protocol:

- Train: SciFact `trainfit`
- Dev selection: SciFact `dev`
- Final diagnostic/frozen evaluation: SciFact `test`
- Components: BM25, BGE-small-final specialist, BGE-base generalist

## Dev Result

| Stage | Method | Switch | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| C:bm25 | Component BM25 | N/A | 0.0662 | -0.8390 | 0.1332 | 0.3973 | 0.0490 |
| C:bge_small | Component BGE-small-final | N/A | 0.9052 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |
| C:bge_base | Component BGE-base | N/A | 0.7234 | -0.1818 | 0.8336 | 0.9907 | 0.6972 |
| O1 | Oracle component router | N/A | 0.9235 | +0.0182 | 0.9938 | 1.0000 | 0.9079 |
| A1 | Multiclass adaptive router | 0.2857 | 0.8655 | -0.0397 | 0.9563 | 1.0000 | 0.8470 |
| A2 | Binary BGE-base rescue gate | 0.0807 | 0.9117 | +0.0065 | 0.9925 | 1.0000 | 0.8887 |
| A3 | Coverage-controlled BGE-base rescue gate | 0.0497 | 0.9098 | +0.0046 | 0.9925 | 1.0000 | 0.8861 |
| A4 | Coverage-controlled BM25 lexical rescue gate | 0.0000 | 0.9052 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |
| A5 | Dual coverage-controlled BGE-base + BM25 rescue gate | 0.0497 | 0.9098 | +0.0046 | 0.9925 | 1.0000 | 0.8861 |

## Test Result

| Stage | Method | Switch | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| C:bm25 | Component BM25 | N/A | 0.6523 | -0.1665 | 0.7757 | 0.8731 | 0.6184 |
| C:bge_small | Component BGE-small-final | N/A | 0.8188 | +0.0000 | 0.9349 | 0.9783 | 0.7875 |
| C:bge_base | Component BGE-base | N/A | 0.7376 | -0.0812 | 0.8659 | 0.9700 | 0.7004 |
| O1 | Oracle component router | N/A | 0.8786 | +0.0598 | 0.9609 | 0.9783 | 0.8563 |
| A1 | Multiclass adaptive router | 0.5833 | 0.7827 | -0.0361 | 0.9149 | 0.9800 | 0.7463 |
| A2 | Binary BGE-base rescue gate | 0.3500 | 0.8015 | -0.0173 | 0.9316 | 0.9817 | 0.7675 |
| A3 | Coverage-controlled BGE-base rescue gate | 0.0500 | 0.8216 | +0.0028 | 0.9382 | 0.9783 | 0.7900 |
| A4 | Coverage-controlled BM25 lexical rescue gate | 0.0000 | 0.8188 | +0.0000 | 0.9349 | 0.9783 | 0.7875 |
| A5 | Dual coverage-controlled BGE-base + BM25 rescue gate | 0.0500 | 0.8216 | +0.0028 | 0.9382 | 0.9783 | 0.7900 |

## Diagnostics

Oracle label distribution:

| Split | BM25 | BGE-small | BGE-base |
|---|---:|---:|---:|
| trainfit | 0 | 615 | 33 |
| dev | 0 | 151 | 10 |
| test | 27 | 251 | 22 |

Contribution cases for A3:

| Split | Changed | Helped | Hurt | Neutral Changed | Missed BGE-base Rescue | Mean Delta |
|---|---:|---:|---:|---:|---:|---:|
| dev | 8 | 3 | 2 | 3 | 7 | +0.0046 |
| test | 15 | 4 | 4 | 7 | 31 | +0.0028 |

Paired bootstrap for A3 vs BGE-small-final:

| Split | Queries | Mean delta nDCG@10 | 95% CI | p-value | Significant |
|---|---:|---:|---:|---:|---|
| dev | 161 | +0.004629 | [-0.003003, +0.013754] | 0.2530 | no |
| test | 300 | +0.002813 | [-0.003648, +0.010259] | 0.4304 | no |

BM25 low-coverage rescue result:

| Stage | Selected signal | Selected BM25 coverage | Dev delta | Test delta | Interpretation |
|---|---|---:|---:|---:|---|
| A4 | bm25_gap_minus_specialist_gap | 0.0000 | +0.0000 | +0.0000 | no clean trainfit BM25-positive signal |
| A5 | A3 + bm25_gap_minus_specialist_gap | 0.0000 | +0.0046 | +0.0028 | identical to A3 under clean selection |

## Interpretation

- A1 fails because a multiclass router over-switches to the weaker generalist.
- A2 passes the dev success criterion but fails on test because absolute probability thresholds are poorly calibrated across splits.
- A3 fixes most of that calibration drift by selecting a fixed coverage (`5%`) of highest-confidence BGE-base rescue queries per split.
- A4/A5 show that BM25 rescue cannot be selected under the clean trainfit protocol: trainfit has `0` BM25 oracle-positive labels, so the best low-coverage lexical gate is no-op.
- A3 is the best current SGAF candidate: small but positive on both dev and test with low switch rate, but not statistically significant under paired bootstrap.

## Decision

Do not claim a large final win yet. The current novelty candidate is:

> Coverage-controlled specialist-generalist rescue: preserve the specialist by default, rank queries by a learned generalist-rescue probability, and route only a fixed low-coverage budget to the generalist.

Next work should test robustness beyond SciFact:

1. Evaluate cross-dataset transfer after the SciFact recipe is frozen.
2. Run duplicate-filtered SciFact sensitivity.
3. Keep BM25 rescue as a negative/diagnostic ablation unless a new clean training signal appears.
