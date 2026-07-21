# SGAF BGE-base Mode-Switch Ablation

This is a Phase 7 diagnostic. It tests whether the BGE-base transfer gap is caused by the hard coverage cap or by the current uncertainty formula being too conservative.

## Method Summary

| Method | Avg nDCG@10 | Transfer Avg | Avg delta vs BGE-small | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|---:|---:|
| Component bm25 | 0.3316 | 0.2247 | -0.0990 | -0.1003 | -0.1665 |
| Component bge_small | 0.4305 | 0.3011 | +0.0000 | -0.0239 | +0.0000 |
| Component bge_base | 0.4282 | 0.3251 | -0.0024 | +0.0000 | -0.0812 |
| Fixed A3 coverage 5% | 0.4325 | 0.3028 | +0.0020 | -0.0222 | +0.0028 |
| Current uncertainty coverage | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 |
| Cap-only max 0.40 | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 |
| Cap-only max 0.60 | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 |
| Cap-only max 0.80 | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 |
| Cap-only max 1.00 | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 |
| Gain 1.5 max 1.00 | 0.4377 | 0.3110 | +0.0071 | -0.0140 | -0.0012 |
| Gain 2.0 max 1.00 | 0.4380 | 0.3128 | +0.0075 | -0.0122 | -0.0051 |
| Gain 3.0 max 1.00 | 0.4397 | 0.3156 | +0.0091 | -0.0094 | -0.0070 |
| Gain 4.0 max 1.00 | 0.4405 | 0.3176 | +0.0100 | -0.0074 | -0.0097 |
| Gain 6.0 max 1.00 | 0.4456 | 0.3249 | +0.0151 | -0.0001 | -0.0111 |
| Gain 8.0 max 1.00 | 0.4459 | 0.3251 | +0.0154 | +0.0000 | -0.0103 |
| Batch shift t2.0 gain 4.0 | 0.4437 | 0.3176 | +0.0131 | -0.0074 | +0.0030 |
| Batch shift t2.0 gain 6.0 | 0.4492 | 0.3249 | +0.0186 | -0.0001 | +0.0030 |
| Batch shift t2.0 gain 8.0 | 0.4492 | 0.3251 | +0.0187 | +0.0000 | +0.0030 |
| Batch shift t3.0 gain 4.0 | 0.4437 | 0.3176 | +0.0131 | -0.0074 | +0.0030 |
| Batch shift t3.0 gain 6.0 | 0.4492 | 0.3249 | +0.0186 | -0.0001 | +0.0030 |
| Batch shift t4.0 gain 6.0 | 0.4411 | 0.3143 | +0.0106 | -0.0108 | +0.0030 |

## Dataset Detail

| Dataset | Stage | Method | Coverage | nDCG@10 | Delta vs BGE-small | Delta vs BGE-base | Delta vs current | Recall@100 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| scifact | B:bge_small | Component bge_small | N/A | 0.8188 | +0.0000 | +0.0812 | -0.0030 | 0.9783 |
| scifact | B:bge_base | Component bge_base | N/A | 0.7376 | -0.0812 | +0.0000 | -0.0842 | 0.9700 |
| scifact | B2 | Fixed A3 coverage 5% | 0.050 | 0.8216 | +0.0028 | +0.0840 | -0.0001 | 0.9783 |
| scifact | B3 | Current uncertainty coverage | 0.084 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| scifact | B4a | Cap-only max 1.00 | 0.084 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| scifact | B4b | Gain 2.0 max 1.00 | 0.117 | 0.8137 | -0.0051 | +0.0761 | -0.0081 | 0.9783 |
| scifact | B4b | Gain 4.0 max 1.00 | 0.185 | 0.8091 | -0.0097 | +0.0715 | -0.0127 | 0.9783 |
| scifact | B4b | Gain 8.0 max 1.00 | 0.319 | 0.8085 | -0.0103 | +0.0709 | -0.0133 | 0.9783 |
| scifact | B5 | Batch shift t2.0 gain 4.0 | 0.084 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| scifact | B5 | Batch shift t2.0 gain 6.0 | 0.084 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| scifact | B5 | Batch shift t2.0 gain 8.0 | 0.084 | 0.8218 | +0.0030 | +0.0842 | +0.0000 | 0.9783 |
| nfcorpus | B:bge_small | Component bge_small | N/A | 0.3505 | +0.0000 | -0.0191 | -0.0055 | 0.3276 |
| nfcorpus | B:bge_base | Component bge_base | N/A | 0.3695 | +0.0191 | +0.0000 | +0.0136 | 0.3320 |
| nfcorpus | B2 | Fixed A3 coverage 5% | 0.050 | 0.3530 | +0.0025 | -0.0166 | -0.0030 | 0.3277 |
| nfcorpus | B3 | Current uncertainty coverage | 0.180 | 0.3559 | +0.0055 | -0.0136 | +0.0000 | 0.3293 |
| nfcorpus | B4a | Cap-only max 1.00 | 0.180 | 0.3559 | +0.0055 | -0.0136 | +0.0000 | 0.3293 |
| nfcorpus | B4b | Gain 2.0 max 1.00 | 0.309 | 0.3579 | +0.0074 | -0.0117 | +0.0019 | 0.3326 |
| nfcorpus | B4b | Gain 4.0 max 1.00 | 0.569 | 0.3598 | +0.0093 | -0.0097 | +0.0039 | 0.3349 |
| nfcorpus | B4b | Gain 8.0 max 1.00 | 1.000 | 0.3695 | +0.0191 | +0.0000 | +0.0136 | 0.3320 |
| nfcorpus | B5 | Batch shift t2.0 gain 4.0 | 0.569 | 0.3598 | +0.0093 | -0.0097 | +0.0039 | 0.3349 |
| nfcorpus | B5 | Batch shift t2.0 gain 6.0 | 0.828 | 0.3692 | +0.0187 | -0.0003 | +0.0133 | 0.3404 |
| nfcorpus | B5 | Batch shift t2.0 gain 8.0 | 1.000 | 0.3695 | +0.0191 | +0.0000 | +0.0136 | 0.3320 |
| fiqa | B:bge_small | Component bge_small | N/A | 0.3635 | +0.0000 | -0.0274 | -0.0138 | 0.6544 |
| fiqa | B:bge_base | Component bge_base | N/A | 0.3909 | +0.0274 | +0.0000 | +0.0136 | 0.7314 |
| fiqa | B2 | Fixed A3 coverage 5% | 0.050 | 0.3653 | +0.0017 | -0.0257 | -0.0121 | 0.6577 |
| fiqa | B3 | Current uncertainty coverage | 0.219 | 0.3774 | +0.0138 | -0.0136 | +0.0000 | 0.6710 |
| fiqa | B4a | Cap-only max 1.00 | 0.219 | 0.3774 | +0.0138 | -0.0136 | +0.0000 | 0.6710 |
| fiqa | B4b | Gain 2.0 max 1.00 | 0.388 | 0.3815 | +0.0180 | -0.0094 | +0.0041 | 0.6846 |
| fiqa | B4b | Gain 4.0 max 1.00 | 0.726 | 0.3855 | +0.0220 | -0.0054 | +0.0081 | 0.7111 |
| fiqa | B4b | Gain 8.0 max 1.00 | 1.000 | 0.3909 | +0.0274 | +0.0000 | +0.0136 | 0.7314 |
| fiqa | B5 | Batch shift t2.0 gain 4.0 | 0.726 | 0.3855 | +0.0220 | -0.0054 | +0.0081 | 0.7111 |
| fiqa | B5 | Batch shift t2.0 gain 6.0 | 1.000 | 0.3909 | +0.0274 | +0.0000 | +0.0136 | 0.7314 |
| fiqa | B5 | Batch shift t2.0 gain 8.0 | 1.000 | 0.3909 | +0.0274 | +0.0000 | +0.0136 | 0.7314 |
| scidocs | B:bge_small | Component bge_small | N/A | 0.1893 | +0.0000 | -0.0253 | -0.0068 | 0.4559 |
| scidocs | B:bge_base | Component bge_base | N/A | 0.2147 | +0.0253 | +0.0000 | +0.0185 | 0.5104 |
| scidocs | B2 | Fixed A3 coverage 5% | 0.050 | 0.1902 | +0.0008 | -0.0245 | -0.0060 | 0.4571 |
| scidocs | B3 | Current uncertainty coverage | 0.211 | 0.1962 | +0.0068 | -0.0185 | +0.0000 | 0.4639 |
| scidocs | B4a | Cap-only max 1.00 | 0.211 | 0.1962 | +0.0068 | -0.0185 | +0.0000 | 0.4639 |
| scidocs | B4b | Gain 2.0 max 1.00 | 0.372 | 0.1991 | +0.0097 | -0.0156 | +0.0029 | 0.4709 |
| scidocs | B4b | Gain 4.0 max 1.00 | 0.695 | 0.2076 | +0.0183 | -0.0071 | +0.0114 | 0.4888 |
| scidocs | B4b | Gain 8.0 max 1.00 | 1.000 | 0.2147 | +0.0253 | +0.0000 | +0.0185 | 0.5104 |
| scidocs | B5 | Batch shift t2.0 gain 4.0 | 0.695 | 0.2076 | +0.0183 | -0.0071 | +0.0114 | 0.4888 |
| scidocs | B5 | Batch shift t2.0 gain 6.0 | 1.000 | 0.2147 | +0.0253 | +0.0000 | +0.0185 | 0.5104 |
| scidocs | B5 | Batch shift t2.0 gain 8.0 | 1.000 | 0.2147 | +0.0253 | +0.0000 | +0.0185 | 0.5104 |

## Significance

| Dataset | Method | Baseline | Mean delta | 95% CI | p-value | Significant |
|---|---|---|---:|---:|---:|---|
| scifact | Current uncertainty coverage | bge_base | +0.084150 | [+0.053150, +0.113815] | 0.0000 | yes |
| scifact | Cap-only max 1.00 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scifact | Cap-only max 1.00 | bge_base | +0.084150 | [+0.055217, +0.114183] | 0.0000 | yes |
| scifact | Gain 2.0 max 1.00 | current adaptive | -0.008082 | [-0.016419, -0.001965] | 0.0140 | yes |
| scifact | Gain 2.0 max 1.00 | bge_base | +0.076068 | [+0.049423, +0.104183] | 0.0000 | yes |
| scifact | Gain 4.0 max 1.00 | current adaptive | -0.012677 | [-0.025895, -0.000920] | 0.0340 | yes |
| scifact | Gain 4.0 max 1.00 | bge_base | +0.071474 | [+0.045100, +0.099125] | 0.0000 | yes |
| scifact | Gain 8.0 max 1.00 | current adaptive | -0.013281 | [-0.027130, +0.002094] | 0.0780 | no |
| scifact | Gain 8.0 max 1.00 | bge_base | +0.070869 | [+0.047613, +0.095599] | 0.0000 | yes |
| scifact | Batch shift t2.0 gain 4.0 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scifact | Batch shift t2.0 gain 4.0 | bge_base | +0.084150 | [+0.056697, +0.113749] | 0.0000 | yes |
| scifact | Batch shift t2.0 gain 6.0 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scifact | Batch shift t2.0 gain 6.0 | bge_base | +0.084150 | [+0.056355, +0.113841] | 0.0000 | yes |
| scifact | Batch shift t2.0 gain 8.0 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scifact | Batch shift t2.0 gain 8.0 | bge_base | +0.084150 | [+0.056041, +0.114618] | 0.0000 | yes |
| nfcorpus | Current uncertainty coverage | bge_base | -0.013590 | [-0.028116, +0.001114] | 0.0720 | no |
| nfcorpus | Cap-only max 1.00 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| nfcorpus | Cap-only max 1.00 | bge_base | -0.013590 | [-0.028660, +0.000757] | 0.0600 | no |
| nfcorpus | Gain 2.0 max 1.00 | current adaptive | +0.001908 | [-0.002750, +0.006505] | 0.4060 | no |
| nfcorpus | Gain 2.0 max 1.00 | bge_base | -0.011682 | [-0.025701, +0.002231] | 0.0980 | no |
| nfcorpus | Gain 4.0 max 1.00 | current adaptive | +0.003854 | [-0.003638, +0.012079] | 0.3420 | no |
| nfcorpus | Gain 4.0 max 1.00 | bge_base | -0.009736 | [-0.022001, +0.002496] | 0.1300 | no |
| nfcorpus | Gain 8.0 max 1.00 | current adaptive | +0.013590 | [+0.000152, +0.027179] | 0.0500 | yes |
| nfcorpus | Gain 8.0 max 1.00 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| nfcorpus | Batch shift t2.0 gain 4.0 | current adaptive | +0.003854 | [-0.003933, +0.012299] | 0.3320 | no |
| nfcorpus | Batch shift t2.0 gain 4.0 | bge_base | -0.009736 | [-0.022313, +0.002626] | 0.1160 | no |
| nfcorpus | Batch shift t2.0 gain 6.0 | current adaptive | +0.013267 | [+0.000276, +0.026827] | 0.0440 | yes |
| nfcorpus | Batch shift t2.0 gain 6.0 | bge_base | -0.000323 | [-0.005056, +0.004791] | 0.8860 | no |
| nfcorpus | Batch shift t2.0 gain 8.0 | current adaptive | +0.013590 | [-0.001286, +0.027409] | 0.0660 | no |
| nfcorpus | Batch shift t2.0 gain 8.0 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| fiqa | Current uncertainty coverage | bge_base | -0.013555 | [-0.027452, +0.001410] | 0.0680 | no |
| fiqa | Cap-only max 1.00 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| fiqa | Cap-only max 1.00 | bge_base | -0.013555 | [-0.027238, +0.001502] | 0.0920 | no |
| fiqa | Gain 2.0 max 1.00 | current adaptive | +0.004142 | [-0.001754, +0.010419] | 0.1740 | no |
| fiqa | Gain 2.0 max 1.00 | bge_base | -0.009413 | [-0.022579, +0.002753] | 0.1500 | no |
| fiqa | Gain 4.0 max 1.00 | current adaptive | +0.008111 | [-0.003513, +0.020093] | 0.1540 | no |
| fiqa | Gain 4.0 max 1.00 | bge_base | -0.005443 | [-0.013499, +0.003461] | 0.2320 | no |
| fiqa | Gain 8.0 max 1.00 | current adaptive | +0.013555 | [+0.000541, +0.028512] | 0.0440 | yes |
| fiqa | Gain 8.0 max 1.00 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| fiqa | Batch shift t2.0 gain 4.0 | current adaptive | +0.008111 | [-0.003297, +0.018816] | 0.1840 | no |
| fiqa | Batch shift t2.0 gain 4.0 | bge_base | -0.005443 | [-0.013333, +0.002813] | 0.1980 | no |
| fiqa | Batch shift t2.0 gain 6.0 | current adaptive | +0.013555 | [-0.000905, +0.028339] | 0.0680 | no |
| fiqa | Batch shift t2.0 gain 6.0 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| fiqa | Batch shift t2.0 gain 8.0 | current adaptive | +0.013555 | [-0.001148, +0.027698] | 0.0640 | no |
| fiqa | Batch shift t2.0 gain 8.0 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scidocs | Current uncertainty coverage | bge_base | -0.018481 | [-0.025939, -0.010318] | 0.0000 | yes |
| scidocs | Cap-only max 1.00 | current adaptive | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scidocs | Cap-only max 1.00 | bge_base | -0.018481 | [-0.026420, -0.010946] | 0.0000 | yes |
| scidocs | Gain 2.0 max 1.00 | current adaptive | +0.002863 | [-0.000518, +0.005988] | 0.0960 | no |
| scidocs | Gain 2.0 max 1.00 | bge_base | -0.015619 | [-0.022942, -0.008755] | 0.0000 | yes |
| scidocs | Gain 4.0 max 1.00 | current adaptive | +0.011418 | [+0.005602, +0.017472] | 0.0000 | yes |
| scidocs | Gain 4.0 max 1.00 | bge_base | -0.007063 | [-0.013052, -0.001545] | 0.0020 | yes |
| scidocs | Gain 8.0 max 1.00 | current adaptive | +0.018481 | [+0.010671, +0.026752] | 0.0000 | yes |
| scidocs | Gain 8.0 max 1.00 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scidocs | Batch shift t2.0 gain 4.0 | current adaptive | +0.011418 | [+0.005890, +0.017305] | 0.0000 | yes |
| scidocs | Batch shift t2.0 gain 4.0 | bge_base | -0.007063 | [-0.012573, -0.001788] | 0.0080 | yes |
| scidocs | Batch shift t2.0 gain 6.0 | current adaptive | +0.018481 | [+0.010423, +0.026465] | 0.0000 | yes |
| scidocs | Batch shift t2.0 gain 6.0 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| scidocs | Batch shift t2.0 gain 8.0 | current adaptive | +0.018481 | [+0.010443, +0.026451] | 0.0000 | yes |
| scidocs | Batch shift t2.0 gain 8.0 | bge_base | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |

## Interpretation

- If cap-only rows equal current uncertainty coverage, the hard cap is not the immediate blocker.
- If gain rows improve transfer but hurt SciFact, Phase 7 should use a mode switch rather than a global gain increase.
- Batch shift rows test that mode-switch hypothesis by keeping source-like batches at current coverage and increasing coverage only for shifted batches.
- Gain rows are exploratory diagnostics; they use target evaluation to understand the failure mode and should not be promoted as a frozen final method.
