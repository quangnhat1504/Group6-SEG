# SGAF P4 Residual Specialist Fallback Ablation

P4 is a cheap post-retrieval diagnostic after Frozen P3. It applies only inside `generalist_fallback` batches and replaces a top-fraction of queries with the BGE-small specialist run according to a label-free BGE-small confidence signal.

LOTO held-out mean delta vs Frozen P3: `+0.0021`.
All LOTO guards pass: `True`. All held-out deltas are positive: `True`.

## Sweep Summary

| Variant | Feature | Fraction | Transfer Avg | Delta vs P3 | Delta vs BGE-base | Min Delta | SciFact Delta | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p4_small_minus_base_top_frac_0_100 | small_minus_base_top | 0.1000 | 0.3314 | +0.0021 | +0.0063 | +0.0016 | +0.0000 | candidate |
| p4_small_minus_base_top_frac_0_150 | small_minus_base_top | 0.1500 | 0.3308 | +0.0015 | +0.0057 | +0.0008 | +0.0000 | candidate |
| p4_small_minus_base_top_frac_0_050 | small_minus_base_top | 0.0500 | 0.3301 | +0.0009 | +0.0051 | +0.0004 | +0.0000 | diagnostic_positive |
| p4_small_minus_base_top_frac_0_300 | small_minus_base_top | 0.3000 | 0.3299 | +0.0007 | +0.0049 | -0.0009 | +0.0000 | diagnostic_positive |
| p4_small_minus_base_top_frac_0_200 | small_minus_base_top | 0.2000 | 0.3297 | +0.0004 | +0.0046 | -0.0031 | +0.0000 | diagnostic_positive |
| p4_small_std10_frac_0_100 | small_std10 | 0.1000 | 0.3296 | +0.0003 | +0.0045 | -0.0007 | +0.0000 | diagnostic_positive |
| p4_small_top_frac_0_050 | small_top | 0.0500 | 0.3295 | +0.0003 | +0.0045 | -0.0004 | +0.0000 | diagnostic_positive |
| p4_small_std10_frac_0_050 | small_std10 | 0.0500 | 0.3294 | +0.0001 | +0.0043 | -0.0013 | +0.0000 | diagnostic_positive |
| p4_small_std10_frac_0_150 | small_std10 | 0.1500 | 0.3292 | -0.0000 | +0.0042 | -0.0016 | +0.0000 | reject |
| p4_small_top_frac_0_100 | small_top | 0.1000 | 0.3286 | -0.0006 | +0.0036 | -0.0023 | +0.0000 | reject |
| p4_small_top_frac_0_150 | small_top | 0.1500 | 0.3284 | -0.0009 | +0.0033 | -0.0025 | +0.0000 | reject |
| p4_small_top_frac_0_200 | small_top | 0.2000 | 0.3277 | -0.0015 | +0.0027 | -0.0043 | +0.0000 | reject |

## Leave-One-Transfer-Dataset-Out

| Held-out | Selected on | Variant | Guard | P3 | P4 | Delta vs P3 | Delta vs BGE-base | Selected Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nfcorpus | fiqa+scidocs | p4_small_minus_base_top_frac_0_100 | yes | 0.3744 | 0.3767 | +0.0023 | +0.0071 | 33 |
| fiqa | nfcorpus+scidocs | p4_small_minus_base_top_frac_0_100 | yes | 0.3960 | 0.3977 | +0.0016 | +0.0067 | 65 |
| scidocs | nfcorpus+fiqa | p4_small_minus_base_top_frac_0_100 | yes | 0.2173 | 0.2198 | +0.0024 | +0.0051 | 100 |

## Paired Bootstrap For LOTO Held-Out

| Held-out | Baseline | Queries | Mean delta | CI low | CI high | p-value | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nfcorpus | Frozen P3 rank-window smoothing SGAF | 323 | +0.0023 | -0.0001 | +0.0049 | 0.0610 | no |
| nfcorpus | BGE-base generalist | 323 | +0.0071 | +0.0002 | +0.0141 | 0.0442 | yes |
| fiqa | Frozen P3 rank-window smoothing SGAF | 648 | +0.0016 | -0.0031 | +0.0072 | 0.5408 | no |
| fiqa | BGE-base generalist | 648 | +0.0067 | -0.0012 | +0.0150 | 0.0990 | no |
| scidocs | Frozen P3 rank-window smoothing SGAF | 1000 | +0.0024 | +0.0001 | +0.0048 | 0.0422 | yes |
| scidocs | BGE-base generalist | 1000 | +0.0051 | +0.0014 | +0.0089 | 0.0076 | yes |

## Interpretation

- P4 is a plausible residual fallback candidate, but it is weaker evidence than P3 until tested on a new external dataset or batch.
- The mechanism is cheap and interpretable: after a generalist fallback, recover a subset of high-confidence specialist queries.
- Keep B5/P3 as the main claim; P4 should remain an optional post-retrieval extension unless external validation confirms it.
