# Frozen P3 Rank-Window Smoothing SGAF Candidate

Frozen recipe: use P3 variant `p3_window_20_alpha_0_100` (`window=20`, `alpha=0.10`, `rrf_k=60`) after Frozen B5, only on `generalist_fallback` batches.

This report is a candidate synthesis from existing Phase 8 artifacts. It does not rerun retrieval and does not reselect hyperparameters.

## Summary

| Method | Avg nDCG@10 | Transfer Avg | Avg delta vs BGE-small | Transfer delta vs BGE-base | Transfer delta vs B5 | SciFact delta |
| --- | --- | --- | --- | --- | --- | --- |
| BGE-small specialist | 0.4305 | 0.3011 | +0.0000 | -0.0239 | -0.0238 | +0.0000 |
| BGE-base generalist | 0.4282 | 0.3251 | -0.0024 | +0.0000 | +0.0001 | -0.0812 |
| Current adaptive SGAF | 0.4378 | 0.3098 | +0.0073 | -0.0152 | -0.0151 | +0.0030 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | +0.0186 | -0.0001 | +0.0000 | +0.0030 |
| Frozen P3 rank-window smoothing SGAF | 0.4524 | 0.3293 | +0.0218 | +0.0042 | +0.0043 | +0.0030 |

## Dataset Detail For P3

| Dataset | B5 mode | P3 nDCG@10 | Delta vs BGE-small | Delta vs BGE-base | Delta vs current | Delta vs B5 | Recall@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | specialist_safe | 0.8218 | +0.0030 | +0.0842 | +0.0000 | +0.0000 | 0.9783 |
| nfcorpus | generalist_fallback | 0.3744 | +0.0239 | +0.0048 | +0.0184 | +0.0052 | 0.3433 |
| fiqa | generalist_fallback | 0.3960 | +0.0325 | +0.0051 | +0.0186 | +0.0051 | 0.7389 |
| scidocs | generalist_fallback | 0.2173 | +0.0280 | +0.0027 | +0.0212 | +0.0027 | 0.5146 |

## Duplicate-Filtered SciFact

| Method | Filtered nDCG@10 | Filtered delta vs BGE-small | Note |
| --- | --- | --- | --- |
| BGE-small specialist | 0.8176 | +0.0000 | measured |
| BGE-base generalist | 0.7409 | -0.0767 | measured |
| Current adaptive SGAF | 0.8206 | +0.0030 | measured |
| Frozen B5 mode-switch SGAF | 0.8206 | +0.0030 | measured |
| Frozen P3 rank-window smoothing SGAF | 0.8206 | +0.0030 | same as Frozen B5; SciFact is specialist_safe and not smoothed |

## LOTO Evidence

| Held-out | Selected variant | Delta vs B5 | Delta vs BGE-base |
| --- | --- | --- | --- |
| nfcorpus | p3_window_20_alpha_0_100 | +0.0052 | +0.0048 |
| fiqa | p3_window_20_alpha_0_100 | +0.0051 | +0.0051 |
| scidocs | p3_window_20_alpha_0_100 | +0.0027 | +0.0027 |

## Paired Bootstrap For LOTO Held-Out

| Held-out | Baseline | Mean delta | CI low | CI high | p-value | Significant |
| --- | --- | --- | --- | --- | --- | --- |
| nfcorpus | Frozen B5 mode-switch SGAF | +0.0052 | 0.0008 | 0.0096 | 0.0218 | True |
| nfcorpus | BGE-base generalist | +0.0048 | -0.0016 | 0.0116 | 0.1414 | False |
| fiqa | Frozen B5 mode-switch SGAF | +0.0051 | -0.0009 | 0.0110 | 0.0938 | False |
| fiqa | BGE-base generalist | +0.0051 | -0.0010 | 0.0112 | 0.1044 | False |
| scidocs | Frozen B5 mode-switch SGAF | +0.0027 | -0.0002 | 0.0055 | 0.0670 | False |
| scidocs | BGE-base generalist | +0.0027 | -0.0001 | 0.0054 | 0.0650 | False |

## Interpretation

- Frozen P3 raises transfer average to `0.3293`, above Frozen B5 and BGE-base on the current evaluated transfer datasets.
- LOTO selection chooses the same variant in all three transfer folds, so the signal is not driven by a single held-out dataset.
- Keep the claim caveated: the P3 grid was designed after seeing the project datasets, so external frozen validation is still required for a paper-grade claim.
