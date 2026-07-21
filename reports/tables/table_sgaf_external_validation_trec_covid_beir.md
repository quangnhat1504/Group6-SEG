# SGAF External Validation: trec_covid_beir

This report scores already-generated external-validation ranking files. It does not tune frozen SGAF parameters.

Manifest: `runs/fusion/external_validation/trec_covid_beir/validation_manifest.json`
Qrels: `data/external/trec_covid_beir/test_qrels.csv`

## Summary

| Method | nDCG@10 | Transfer Avg | Recall@10 | Recall@100 | MRR@10 | Cost guard |
|---|---:|---:|---:|---:|---:|---|
| BM25 | 0.5607 | 0.5607 | 0.0153 | 0.0936 | 0.7580 | negligible |
| BGE-small specialist | 0.6185 | 0.6185 | 0.0177 | 0.1136 | 0.8532 | negligible |
| BGE-base generalist | 0.6835 | 0.6835 | 0.0200 | 0.1340 | 0.8545 | negligible |
| Current adaptive SGAF | 0.6304 | 0.6304 | 0.0182 | 0.1157 | 0.8432 | negligible |
| Frozen B5 mode-switch SGAF | 0.6835 | 0.6835 | 0.0200 | 0.1340 | 0.8545 | negligible |
| Frozen P3 rank-window smoothing SGAF | 0.6908 | 0.6908 | 0.0203 | 0.1336 | 0.8367 | negligible |
| Optional P4 residual specialist fallback | 0.6837 | 0.6837 | 0.0202 | 0.1318 | 0.8500 | negligible |

## Contribution Rows

| Method | nDCG@10 | Delta vs BGE-base | Delta vs B5 | Delta vs P3 | Run path |
|---|---:|---:|---:|---:|---|
| BM25 | 0.5607 | -0.1227 | -0.1227 | -0.1301 | `runs/fusion/external_validation/trec_covid_beir/bm25.csv` |
| BGE-small specialist | 0.6185 | -0.0650 | -0.0650 | -0.0723 | `runs/fusion/external_validation/trec_covid_beir/bge_small_specialist.csv` |
| BGE-base generalist | 0.6835 | +0.0000 | +0.0000 | -0.0073 | `runs/fusion/external_validation/trec_covid_beir/bge_base_generalist.csv` |
| Current adaptive SGAF | 0.6304 | -0.0530 | -0.0530 | -0.0604 | `runs/fusion/external_validation/trec_covid_beir/current_adaptive_sgaf.csv` |
| Frozen B5 mode-switch SGAF | 0.6835 | +0.0000 | +0.0000 | -0.0073 | `runs/fusion/external_validation/trec_covid_beir/frozen_b5_sgaf.csv` |
| Frozen P3 rank-window smoothing SGAF | 0.6908 | +0.0073 | +0.0073 | +0.0000 | `runs/fusion/external_validation/trec_covid_beir/frozen_p3_sgaf.csv` |
| Optional P4 residual specialist fallback | 0.6837 | +0.0002 | +0.0002 | -0.0071 | `runs/fusion/external_validation/trec_covid_beir/optional_p4_sgaf.csv` |

## Significance

| System | Baseline | Queries | Mean delta | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---:|---|
| Frozen B5 mode-switch SGAF | BGE-base generalist | 50 | +0.0000 | [+0.0000, +0.0000] | 1.0000 | False |
| Frozen P3 rank-window smoothing SGAF | Frozen B5 mode-switch SGAF | 50 | +0.0073 | [-0.0115, +0.0271] | 0.4654 | False |
| Frozen P3 rank-window smoothing SGAF | BGE-base generalist | 50 | +0.0073 | [-0.0113, +0.0267] | 0.4516 | False |
| Optional P4 residual specialist fallback | Frozen P3 rank-window smoothing SGAF | 50 | -0.0071 | [-0.0242, +0.0105] | 0.4140 | False |
| Optional P4 residual specialist fallback | BGE-base generalist | 50 | +0.0002 | [-0.0264, +0.0257] | 1.0000 | False |

## Next Gate

Run:

```powershell
python scripts\audit_sgaf_external_validation_results.py runs\\fusion\\external_validation\\trec_covid_beir\\validation_manifest.json
```
