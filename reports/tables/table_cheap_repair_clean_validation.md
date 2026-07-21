# Cheap Repair Clean Validation

Date: 2026-07-20

Protocol: tune cheap repair on SciFact `trainfit`, evaluate on held-out `dev`.

Artifacts:

- Trainfit specialist run: `runs/scifact/trainfit_dense_bge_small_scifact_rrf.csv`
- Trainfit specialist metrics: `runs/scifact/trainfit_dense_bge_small_scifact_rrf_metrics.json`
- Clean validation CSV: `runs/fusion/cheap_repair_clean_validation/trainfit_to_dev_cheap_repair_clean_validation.csv`
- Failure analysis: `runs/fusion/cheap_repair_clean_validation/dev_cheap_repair_failure_analysis.md`

## Result

| Stage | Method | Selected On | Switch | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S1 | BGE-small-final specialist | fixed | N/A | 0.9052 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |
| S8 | Rank-local BM25 promotion | trainfit | 0.0124 | 0.9052 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |
| S9 | Learned cheap gate + rank-local promotion | trainfit | 0.0124 | 0.9057 | +0.0005 | 0.9938 | 1.0000 | 0.8799 |

## Rescue Headroom

- Trainfit BM25 rescue-positive queries: `2`.
- Dev BM25 rescue-positive queries: `0`.
- Dev S9 helped `1` query, hurt `0`, and missed `0` BM25 rescue queries.

## Decision

S9 is not strong enough to freeze. It improves dev by only `+0.0005` nDCG@10, below the predefined `+0.005` threshold.

Cheap BM25 repair should now be treated as an analyzed diagnostic path, not the main performance direction. Move to Specialist-Generalist Adaptive Fusion or an explicitly separated expensive fallback phase.
