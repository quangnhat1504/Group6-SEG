# Adaptive Coverage SGAF Ablation

The BGE-base rescue ranking model is frozen from SciFact `trainfit`; this phase changes only the coverage budget.

| Dataset | Method | Coverage | nDCG@10 | Delta vs BGE-small | Delta vs Fixed A3 | Recall@100 |
|---|---|---:|---:|---:|---:|---:|
| scifact | Component bge_small | N/A | 0.8188 | +0.0000 | -0.0028 | 0.9783 |
| scifact | Component bge_base | N/A | 0.7376 | -0.0812 | -0.0840 | 0.9700 |
| scifact | Fixed A3 coverage 5% | 0.050 | 0.8216 | +0.0028 | +0.0000 | 0.9783 |
| scifact | Source-shift adaptive coverage | 0.084 | 0.8218 | +0.0030 | +0.0001 | 0.9783 |
| scifact | Uncertainty-shift adaptive coverage | 0.084 | 0.8218 | +0.0030 | +0.0001 | 0.9783 |
| scifact | Conservative-shift adaptive coverage | 0.060 | 0.8225 | +0.0037 | +0.0009 | 0.9783 |
| scifact | Oracle best coverage sweep | 0.050 | 0.8216 | +0.0028 | +0.0000 | 0.9783 |
| scifact | Oracle component router | N/A | 0.8786 | +0.0598 | +0.0570 | 0.9783 |
| nfcorpus | Component bge_small | N/A | 0.3505 | +0.0000 | -0.0025 | 0.3276 |
| nfcorpus | Component bge_base | N/A | 0.3695 | +0.0191 | +0.0166 | 0.3320 |
| nfcorpus | Fixed A3 coverage 5% | 0.050 | 0.3530 | +0.0025 | +0.0000 | 0.3277 |
| nfcorpus | Source-shift adaptive coverage | 0.366 | 0.3578 | +0.0074 | +0.0049 | 0.3337 |
| nfcorpus | Uncertainty-shift adaptive coverage | 0.180 | 0.3559 | +0.0055 | +0.0030 | 0.3293 |
| nfcorpus | Conservative-shift adaptive coverage | 0.144 | 0.3554 | +0.0049 | +0.0024 | 0.3283 |
| nfcorpus | Oracle best coverage sweep | 1.000 | 0.3695 | +0.0191 | +0.0166 | 0.3320 |
| nfcorpus | Oracle component router | N/A | 0.4249 | +0.0744 | +0.0719 | 0.3381 |
| fiqa | Component bge_small | N/A | 0.3635 | +0.0000 | -0.0017 | 0.6544 |
| fiqa | Component bge_base | N/A | 0.3909 | +0.0274 | +0.0257 | 0.7314 |
| fiqa | Fixed A3 coverage 5% | 0.050 | 0.3653 | +0.0017 | +0.0000 | 0.6577 |
| fiqa | Source-shift adaptive coverage | 0.138 | 0.3718 | +0.0082 | +0.0065 | 0.6634 |
| fiqa | Uncertainty-shift adaptive coverage | 0.219 | 0.3774 | +0.0138 | +0.0121 | 0.6710 |
| fiqa | Conservative-shift adaptive coverage | 0.103 | 0.3681 | +0.0045 | +0.0028 | 0.6591 |
| fiqa | Oracle best coverage sweep | 1.000 | 0.3909 | +0.0274 | +0.0257 | 0.7314 |
| fiqa | Oracle component router | N/A | 0.4650 | +0.1014 | +0.0997 | 0.6851 |
| scidocs | Component bge_small | N/A | 0.1893 | +0.0000 | -0.0008 | 0.4559 |
| scidocs | Component bge_base | N/A | 0.2147 | +0.0253 | +0.0245 | 0.5104 |
| scidocs | Fixed A3 coverage 5% | 0.050 | 0.1902 | +0.0008 | +0.0000 | 0.4571 |
| scidocs | Source-shift adaptive coverage | 0.162 | 0.1951 | +0.0058 | +0.0049 | 0.4622 |
| scidocs | Uncertainty-shift adaptive coverage | 0.211 | 0.1962 | +0.0068 | +0.0060 | 0.4639 |
| scidocs | Conservative-shift adaptive coverage | 0.110 | 0.1916 | +0.0022 | +0.0014 | 0.4604 |
| scidocs | Oracle best coverage sweep | 1.000 | 0.2147 | +0.0253 | +0.0245 | 0.5104 |
| scidocs | Oracle component router | N/A | 0.2666 | +0.0773 | +0.0765 | 0.4749 |

## Significance

| Dataset | Comparison | Mean delta | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---|
| scifact | Fixed A3 vs BGE-small | +0.002813 | [-0.003648, +0.010259] | 0.4304 | no |
| scifact | Source-shift adaptive vs fixed A3 | +0.000141 | [-0.005602, +0.005001] | 0.8988 | no |
| scifact | Uncertainty-shift adaptive vs fixed A3 | +0.000141 | [-0.005858, +0.005001] | 0.8888 | no |
| scifact | Conservative-shift adaptive vs fixed A3 | +0.000866 | [+0.000000, +0.002598] | 0.7402 | no |
| nfcorpus | Fixed A3 vs BGE-small | +0.002485 | [-0.000045, +0.005553] | 0.0542 | no |
| nfcorpus | Source-shift adaptive vs fixed A3 | +0.004880 | [-0.002828, +0.012944] | 0.2178 | no |
| nfcorpus | Uncertainty-shift adaptive vs fixed A3 | +0.002979 | [-0.002920, +0.009344] | 0.3286 | no |
| nfcorpus | Conservative-shift adaptive vs fixed A3 | +0.002404 | [-0.003238, +0.008503] | 0.4052 | no |
| fiqa | Fixed A3 vs BGE-small | +0.001748 | [-0.001029, +0.004555] | 0.2274 | no |
| fiqa | Source-shift adaptive vs fixed A3 | +0.006494 | [+0.001891, +0.011556] | 0.0016 | yes |
| fiqa | Uncertainty-shift adaptive vs fixed A3 | +0.012096 | [+0.004609, +0.020205] | 0.0010 | yes |
| fiqa | Conservative-shift adaptive vs fixed A3 | +0.002758 | [-0.000458, +0.006529] | 0.0992 | no |
| scidocs | Fixed A3 vs BGE-small | +0.000834 | [-0.000727, +0.002525] | 0.2962 | no |
| scidocs | Source-shift adaptive vs fixed A3 | +0.004916 | [+0.001860, +0.008071] | 0.0020 | yes |
| scidocs | Uncertainty-shift adaptive vs fixed A3 | +0.006014 | [+0.002514, +0.009626] | 0.0004 | yes |
| scidocs | Conservative-shift adaptive vs fixed A3 | +0.001380 | [-0.000722, +0.003529] | 0.2014 | no |

## Interpretation

- Source-shift coverage is a cheap label-free controller that increases BGE-base coverage when target query/run statistics drift from SciFact trainfit.
- Oracle coverage sweep is diagnostic only; it shows how much budget adaptation could matter if coverage were chosen perfectly per target.
- The key risk is preserving SciFact: overly aggressive coverage hurts SciFact because BGE-base is globally weaker there.
