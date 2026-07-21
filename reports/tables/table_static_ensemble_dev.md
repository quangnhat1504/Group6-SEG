# Static Ensemble Dev Validation

Date: 2026-07-20

Protocol: tune static weighted RRF on SciFact `dev` using existing component runs.

Components:

- `bm25`: `runs/scifact/dev_bm25.csv`
- `bge_small`: `runs/scifact/dev_dense_bge_small_scifact_rrf.csv`
- `bge_base`: `runs/scifact/dev_dense_bge_base.csv`

## Component Metrics

| Component | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Strict Rescue vs BGE-small |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.0662 | 0.1332 | 0.3973 | 0.0490 | 0 |
| bge_small | 0.9052 | 0.9938 | 1.0000 | 0.8799 | 0 |
| bge_base | 0.7234 | 0.8336 | 0.9907 | 0.6972 | 10 |

## Best Static Weighted RRF

| Method | RRF k | Weights | nDCG@10 | Delta vs BGE-small | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---|---:|---:|---:|---:|---:|
| weighted_rrf | 5 | `{"bge_base": 0.0, "bge_small": 2.0, "bm25": 0.25}` | 0.9053 | +0.0000 | 0.9938 | 1.0000 | 0.8799 |

## Diagnostics

- Oracle over `bm25`, `bge_small`, and `bge_base`: `0.9235` nDCG@10.
- Oracle headroom over BGE-small: `+0.0182`.
- BGE-base strictly beats BGE-small on `10` dev queries, but static RRF gives BGE-base zero weight in the best config.

## Decision

Static ensemble does not produce a meaningful dev improvement. The useful signal is the oracle headroom and BGE-base rescue subset, which supports moving to query-adaptive specialist/generalist fusion instead of static global weighting.
