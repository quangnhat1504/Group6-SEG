# QPP Validation: Train vs Test Split Correlations

Train split: 809 queries | Test split: see `qpp_correlations.csv`

| Feature | Train Kendall vs gain | Test Kendall vs gain | Note |
|---|---:|---:|---|
| hybrid_max | -0.1841 | -0.1663 | ★ Top on both splits |
| bm25_dense_overlap | -0.1665 | -0.0291 |  |
| disagreement | +0.1665 | +0.0291 |  |
| hybrid_std | -0.1645 | -0.0454 |  |
| hybrid_nqc | -0.1645 | -0.0454 |  |
| hybrid_wig | -0.1260 | -0.0098 |  |
| bm25_wig | +0.1062 | -0.0049 |  |
| bm25_nqc | +0.0989 | +0.0081 |  |
| bm25_std | +0.0989 | +0.0081 |  |
| bm25_max | +0.0968 | -0.0115 |  |
| hybrid_gap | -0.0670 | -0.0600 |  |
| dense_wig | +0.0369 | +0.0275 |  |
| dense_max | +0.0263 | -0.0305 |  |
| bm25_gap | -0.0116 | +0.0118 |  |
| dense_nqc | +0.0027 | -0.0903 |  |
| dense_std | +0.0027 | -0.0903 |  |
| dense_gap | +0.0006 | +0.0004 |  |

## Summary

- **Train top predictor**: hybrid_max (|Kendall| = 0.1841)
- **Test top predictor**: hybrid_max (|Kendall| = 0.1663)
- **Conclusion**: hybrid_max is the best predictor on BOTH splits → no data leakage in feature selection.
