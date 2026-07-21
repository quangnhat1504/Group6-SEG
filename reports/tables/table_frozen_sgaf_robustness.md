# Frozen SGAF Robustness

Protocol: frozen SciFact A3 gate, selected on `trainfit` only (`C=0.1`, coverage `0.05`).

## Duplicate-Filtered SciFact

| Method | Full nDCG@10 | Filtered nDCG@10 | Filtered delta vs BGE-small | Full Recall@10 | Filtered Recall@10 |
|---|---:|---:|---:|---:|---:|
| BGE-small-final | 0.8188 | 0.8176 | +0.0000 | 0.9349 | 0.9345 |
| Frozen A3 SGAF | 0.8216 | 0.8204 | +0.0028 | 0.9382 | 0.9378 |

## Cross-Dataset Transfer

| Dataset | BGE-small | BGE-base | Oracle | Frozen A3 | A3 delta | A3 switch | Oracle headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| scifact | 0.8188 | 0.7376 | 0.8786 | 0.8216 | +0.0028 | 0.0500 | +0.0598 |
| nfcorpus | 0.3505 | 0.3695 | 0.4249 | 0.3530 | +0.0025 | 0.0495 | +0.0744 |
| fiqa | 0.3635 | 0.3909 | 0.4650 | 0.3653 | +0.0017 | 0.0494 | +0.1014 |
| scidocs | 0.1893 | 0.2147 | 0.2666 | 0.1902 | +0.0008 | 0.0500 | +0.0773 |

## Paired Bootstrap

| Dataset | Queries | Mean delta | 95% CI | p-value | Significant |
|---|---:|---:|---:|---:|---|
| scifact | 300 | +0.002813 | [-0.003648, +0.010259] | 0.4304 | no |
| nfcorpus | 323 | +0.002485 | [+0.000006, +0.005681] | 0.0488 | yes |
| fiqa | 648 | +0.001748 | [-0.001068, +0.004651] | 0.2234 | no |
| scidocs | 1000 | +0.000834 | [-0.000684, +0.002552] | 0.3068 | no |

## Interpretation

- Frozen A3 is directionally positive on all four datasets, but the deltas are small.
- The 5% coverage cap is robust but conservative: BGE-base is globally stronger than BGE-small on NFCorpus, FiQA, and SciDocs, while frozen A3 routes only 5% of queries to BGE-base.
- The oracle headroom remains large, so the next improvement should be coverage adaptation under a clean validation protocol, not BM25 rescue.
