# SGAF Phase 8C Rank-Window Smoothing Ablation

This cheap post-retrieval ablation blends a small BGE-small specialist prior into Frozen B5 only when the batch is already in `generalist_fallback` mode. Source-like SciFact is left unchanged.

Important caveat: this is an exploratory sweep over `window` and `alpha`; do not treat the best row as a frozen final recipe without a new validation split.

## Summary

| Variant | Window | Alpha | Transfer Avg | Delta vs B5 | Delta vs BGE-base | Min transfer delta | SciFact delta | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3_window_20_alpha_0_100 | 20 | 0.1000 | 0.3293 | +0.0043 | +0.0042 | +0.0027 | +0.0000 | candidate |
| p3_window_20_alpha_0_150 | 20 | 0.1500 | 0.3283 | +0.0033 | +0.0032 | +0.0011 | +0.0000 | candidate |
| p3_window_20_alpha_0_050 | 20 | 0.0500 | 0.3282 | +0.0032 | +0.0031 | +0.0021 | +0.0000 | candidate |
| p3_window_20_alpha_0_200 | 20 | 0.2000 | 0.3274 | +0.0025 | +0.0024 | +0.0004 | +0.0000 | candidate |
| p3_window_20_alpha_0_025 | 20 | 0.0250 | 0.3270 | +0.0021 | +0.0020 | +0.0017 | +0.0000 | candidate |
| p3_window_10_alpha_0_050 | 10 | 0.0500 | 0.3262 | +0.0013 | +0.0011 | -0.0009 | +0.0000 | diagnostic_positive |
| p3_window_10_alpha_0_150 | 10 | 0.1500 | 0.3261 | +0.0011 | +0.0010 | -0.0018 | +0.0000 | diagnostic_positive |
| p3_window_10_alpha_0_100 | 10 | 0.1000 | 0.3258 | +0.0009 | +0.0008 | -0.0024 | +0.0000 | diagnostic_positive |
| p3_window_10_alpha_0_025 | 10 | 0.0250 | 0.3258 | +0.0008 | +0.0007 | +0.0001 | +0.0000 | diagnostic_positive |
| p3_window_10_alpha_0_200 | 10 | 0.2000 | 0.3253 | +0.0003 | +0.0002 | -0.0024 | +0.0000 | diagnostic_positive |

## Best Variant Detail

| Dataset | B5 mode | B5 nDCG@10 | Smoothed nDCG@10 | Delta vs B5 | Delta vs BGE-base | Delta vs current |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | specialist_safe | 0.8218 | 0.8218 | +0.0000 | +0.0842 | +0.0000 |
| nfcorpus | generalist_fallback | 0.3692 | 0.3744 | +0.0052 | +0.0048 | +0.0184 |
| fiqa | generalist_fallback | 0.3909 | 0.3960 | +0.0051 | +0.0051 | +0.0186 |
| scidocs | generalist_fallback | 0.2147 | 0.2173 | +0.0027 | +0.0027 | +0.0212 |

## Best Variant Paired Bootstrap

| Dataset | Baseline | Queries | Mean delta | CI low | CI high | p-value | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | Frozen B5 mode-switch SGAF | 300 | +0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| scifact | BGE-base generalist | 300 | +0.0842 | 0.0553 | 0.1143 | 0.0000 | True |
| nfcorpus | Frozen B5 mode-switch SGAF | 323 | +0.0052 | 0.0007 | 0.0097 | 0.0226 | True |
| nfcorpus | BGE-base generalist | 323 | +0.0048 | -0.0016 | 0.0116 | 0.1420 | False |
| fiqa | Frozen B5 mode-switch SGAF | 648 | +0.0051 | -0.0009 | 0.0111 | 0.0958 | False |
| fiqa | BGE-base generalist | 648 | +0.0051 | -0.0010 | 0.0111 | 0.1044 | False |
| scidocs | Frozen B5 mode-switch SGAF | 1000 | +0.0027 | -0.0002 | 0.0055 | 0.0678 | False |
| scidocs | BGE-base generalist | 1000 | +0.0027 | -0.0002 | 0.0054 | 0.0658 | False |

## Interpretation

- A candidate row must improve transfer average by at least `0.001`, avoid any transfer-dataset loss below `-0.0005`, and keep SciFact unchanged.
- If all rows are rejected, adding the specialist prior after Frozen B5 mostly reintroduces the transfer weakness that B5 was designed to avoid.
- If a row is only `diagnostic_positive`, treat it as potential failure-analysis material, not a final method.
