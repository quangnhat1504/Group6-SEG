# SGAF External Validation: arguana

This report scores already-generated external-validation ranking files. It does not tune frozen SGAF parameters.

Manifest: `runs/fusion/external_validation/arguana/validation_manifest.json`
Qrels: `data/external/arguana/test_qrels.csv`

## Summary

| Method | nDCG@10 | Transfer Avg | Recall@10 | Recall@100 | MRR@10 | Cost guard |
|---|---:|---:|---:|---:|---:|---|
| BM25 | 0.3864 | 0.3864 | 0.7952 | 0.9751 | 0.2567 | negligible |
| BGE-small specialist | 0.4222 | 0.4222 | 0.8314 | 0.9865 | 0.2903 | negligible |
| BGE-base generalist | 0.4530 | 0.4530 | 0.8698 | 0.9915 | 0.3175 | negligible |
| Current adaptive SGAF | 0.4263 | 0.4263 | 0.8378 | 0.9865 | 0.2936 | negligible |
| Frozen B5 mode-switch SGAF | 0.4301 | 0.4301 | 0.8428 | 0.9879 | 0.2968 | negligible |
| Frozen P3 rank-window smoothing SGAF | 0.4312 | 0.4312 | 0.8464 | 0.9872 | 0.2973 | negligible |

## Contribution Rows

| Method | nDCG@10 | Delta vs BGE-base | Delta vs B5 | Delta vs P3 | Run path |
|---|---:|---:|---:|---:|---|
| BM25 | 0.3864 | -0.0666 | -0.0436 | -0.0448 | `runs/fusion/external_validation/arguana/bm25.csv` |
| BGE-small specialist | 0.4222 | -0.0309 | -0.0079 | -0.0090 | `runs/fusion/external_validation/arguana/bge_small_specialist.csv` |
| BGE-base generalist | 0.4530 | +0.0000 | +0.0230 | +0.0218 | `runs/fusion/external_validation/arguana/bge_base_generalist.csv` |
| Current adaptive SGAF | 0.4263 | -0.0267 | -0.0037 | -0.0049 | `runs/fusion/external_validation/arguana/current_adaptive_sgaf.csv` |
| Frozen B5 mode-switch SGAF | 0.4301 | -0.0230 | +0.0000 | -0.0011 | `runs/fusion/external_validation/arguana/frozen_b5_sgaf.csv` |
| Frozen P3 rank-window smoothing SGAF | 0.4312 | -0.0218 | +0.0011 | +0.0000 | `runs/fusion/external_validation/arguana/frozen_p3_sgaf.csv` |

## Significance

| System | Baseline | Queries | Mean delta | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---:|---|
| Frozen B5 mode-switch SGAF | BGE-base generalist | 1406 | -0.0230 | [-0.0318, -0.0144] | 0.0000 | True |
| Frozen P3 rank-window smoothing SGAF | Frozen B5 mode-switch SGAF | 1406 | +0.0011 | [+0.0001, +0.0023] | 0.0204 | True |
| Frozen P3 rank-window smoothing SGAF | BGE-base generalist | 1406 | -0.0218 | [-0.0305, -0.0131] | 0.0000 | True |

## Next Gate

Run:

```powershell
python scripts\audit_sgaf_external_validation_results.py runs\\fusion\\external_validation\\arguana\\validation_manifest.json
```
