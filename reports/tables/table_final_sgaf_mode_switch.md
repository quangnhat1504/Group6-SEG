# Final B5 SGAF Mode-Switch Candidate

Frozen recipe: SciFact trainfit BGE-base rescue classifier (`C=0.1`), batch shift threshold `2.0`, shifted uncertainty gain `6.0`, shifted cap `1.0`.

## Summary

| Method | Avg nDCG@10 | Transfer Avg | Avg delta vs BGE-small | Transfer delta vs BGE-base | SciFact delta vs BGE-small | SciFact delta vs current |
|---|---:|---:|---:|---:|---:|---:|
| BGE-small specialist | 0.4305 | 0.3011 | +0.0000 | -0.0239 | +0.0000 | -0.0030 |
| BGE-base generalist | 0.4282 | 0.3251 | -0.0024 | +0.0000 | -0.0812 | -0.0842 |
| Current adaptive SGAF | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 | +0.0000 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | +0.0186 | -0.0001 | +0.0030 | +0.0000 |

## Dataset Detail

| Dataset | Method | Coverage | Mode | Shift | nDCG@10 | Delta vs BGE-small | Delta vs BGE-base | Delta vs current | Recall@100 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| scifact | BGE-small specialist | N/A | component | 1.087 | 0.8188 | +0.0000 | +0.0812 | -0.0030 | 0.9783 |
| scifact | BGE-base generalist | N/A | component | 1.087 | 0.7376 | -0.0812 | +0.0000 | -0.0842 | 0.9700 |
| scifact | Current adaptive SGAF | 0.084 | current_uncertainty | 1.087 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| scifact | Frozen B5 mode-switch SGAF | 0.084 | specialist_safe | 1.087 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| nfcorpus | BGE-small specialist | N/A | component | 6.025 | 0.3505 | +0.0000 | -0.0191 | -0.0055 | 0.3276 |
| nfcorpus | BGE-base generalist | N/A | component | 6.025 | 0.3695 | +0.0191 | +0.0000 | +0.0136 | 0.3320 |
| nfcorpus | Current adaptive SGAF | 0.180 | current_uncertainty | 6.025 | 0.3559 | +0.0055 | -0.0136 | +0.0000 | 0.3293 |
| nfcorpus | Frozen B5 mode-switch SGAF | 0.828 | generalist_fallback | 6.025 | 0.3692 | +0.0187 | -0.0003 | +0.0133 | 0.3404 |
| fiqa | BGE-small specialist | N/A | component | 3.793 | 0.3635 | +0.0000 | -0.0274 | -0.0138 | 0.6544 |
| fiqa | BGE-base generalist | N/A | component | 3.793 | 0.3909 | +0.0274 | +0.0000 | +0.0136 | 0.7314 |
| fiqa | Current adaptive SGAF | 0.219 | current_uncertainty | 3.793 | 0.3774 | +0.0138 | -0.0136 | +0.0000 | 0.6710 |
| fiqa | Frozen B5 mode-switch SGAF | 1.000 | generalist_fallback | 3.793 | 0.3909 | +0.0274 | +0.0000 | +0.0136 | 0.7314 |
| scidocs | BGE-small specialist | N/A | component | 3.584 | 0.1893 | +0.0000 | -0.0253 | -0.0068 | 0.4559 |
| scidocs | BGE-base generalist | N/A | component | 3.584 | 0.2147 | +0.0253 | +0.0000 | +0.0185 | 0.5104 |
| scidocs | Current adaptive SGAF | 0.211 | current_uncertainty | 3.584 | 0.1962 | +0.0068 | -0.0185 | +0.0000 | 0.4639 |
| scidocs | Frozen B5 mode-switch SGAF | 1.000 | generalist_fallback | 3.584 | 0.2147 | +0.0253 | +0.0000 | +0.0185 | 0.5104 |

## Duplicate-Filtered SciFact

| Method | Full nDCG@10 | Filtered nDCG@10 | Filtered delta vs BGE-small | Filtered delta vs current | Full Recall@10 | Filtered Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| BGE-small specialist | 0.8188 | 0.8176 | +0.0000 | -0.0030 | 0.9349 | 0.9345 |
| BGE-base generalist | 0.7376 | 0.7409 | -0.0767 | -0.0797 | 0.8659 | 0.8683 |
| Current adaptive SGAF | 0.8218 | 0.8206 | +0.0030 | +0.0000 | 0.9382 | 0.9378 |
| Frozen B5 mode-switch SGAF | 0.8218 | 0.8206 | +0.0030 | +0.0000 | 0.9382 | 0.9378 |

## Paired Bootstrap

| Dataset | Baseline | Mean delta | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---|
| scifact | Current adaptive SGAF | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scifact | BGE-small specialist | +0.002955 | [-0.005719, +0.011935] | 0.5114 | no |
| scifact | BGE-base generalist | +0.084150 | [+0.055194, +0.114215] | 0.0000 | yes |
| nfcorpus | Current adaptive SGAF | +0.013267 | [-0.000088, +0.027252] | 0.0510 | no |
| nfcorpus | BGE-small specialist | +0.018731 | [+0.003866, +0.034561] | 0.0134 | yes |
| nfcorpus | BGE-base generalist | -0.000323 | [-0.005133, +0.004672] | 0.8784 | no |
| fiqa | Current adaptive SGAF | +0.013555 | [-0.001059, +0.027949] | 0.0692 | no |
| fiqa | BGE-small specialist | +0.027398 | [+0.010672, +0.043733] | 0.0014 | yes |
| fiqa | BGE-base generalist | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scidocs | Current adaptive SGAF | +0.018481 | [+0.010483, +0.026447] | 0.0000 | yes |
| scidocs | BGE-small specialist | +0.025329 | [+0.016543, +0.034359] | 0.0000 | yes |
| scidocs | BGE-base generalist | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |

## Interpretation

- Frozen B5 keeps the current SciFact adaptive score because SciFact is source-like under the batch shift score.
- On shifted datasets, the controller increases BGE-base coverage and nearly matches the BGE-base transfer average.
- This is now the frozen Phase 7 candidate. Because threshold/gain were identified in the exploratory B4/B5 sweep, the paper-grade next step is to keep these values fixed before any new held-out batch or dataset, not to retune them again.
