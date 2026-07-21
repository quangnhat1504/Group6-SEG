# SGAF P3 Leave-One-Transfer-Dataset-Out Validation

For each held-out transfer dataset, the P3 smoothing variant is selected using the other two transfer datasets only.

Held-out mean delta vs Frozen B5: `+0.0043`.
Same selected variant across folds: `True`.
All held-out deltas positive: `True`.

## LOTO Selection

| Held-out | Selected on | Variant | Train mean delta | Train min delta | B5 | P3 | Held-out delta vs B5 | Held-out delta vs BGE-base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nfcorpus | fiqa,scidocs | p3_window_20_alpha_0_100 | +0.0039 | +0.0027 | 0.3692 | 0.3744 | +0.0052 | +0.0048 |
| fiqa | nfcorpus,scidocs | p3_window_20_alpha_0_100 | +0.0039 | +0.0027 | 0.3909 | 0.3960 | +0.0051 | +0.0051 |
| scidocs | nfcorpus,fiqa | p3_window_20_alpha_0_100 | +0.0051 | +0.0051 | 0.2147 | 0.2173 | +0.0027 | +0.0027 |

## Held-Out Paired Bootstrap

| Held-out | Baseline | Queries | Mean delta | CI low | CI high | p-value | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nfcorpus | Frozen B5 mode-switch SGAF | 323 | +0.0052 | +0.0008 | +0.0096 | 0.0218 | True |
| nfcorpus | BGE-base generalist | 323 | +0.0048 | -0.0016 | +0.0116 | 0.1414 | False |
| fiqa | Frozen B5 mode-switch SGAF | 648 | +0.0051 | -0.0009 | +0.0110 | 0.0938 | False |
| fiqa | BGE-base generalist | 648 | +0.0051 | -0.0010 | +0.0112 | 0.1044 | False |
| scidocs | Frozen B5 mode-switch SGAF | 1000 | +0.0027 | -0.0002 | +0.0055 | 0.0670 | False |
| scidocs | BGE-base generalist | 1000 | +0.0027 | -0.0001 | +0.0054 | 0.0650 | False |

## Interpretation

- If the same variant is selected across folds and all held-out deltas are positive, P3 is more than a one-dataset artifact.
- This still is not equivalent to a new dataset validation because the variant grid was designed after seeing the project datasets.
- A stronger next step is to freeze the selected variant and evaluate on a new batch/dataset without changing the grid.
