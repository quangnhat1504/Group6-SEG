---
title: "Paired T-Test"
type: concept
tags: [statistics, significance, t-test]
last_updated: 2026-07-11
---

## What

Paired t-test is the significance testing method used in this project, following Al-Joofi et al. (2025). It measures whether the difference between two retrieval systems is statistically reliable given per-query nDCG@10 scores.

- **Test**: Paired t-test with Shapiro-Wilk normality check
- **Fallback**: Wilcoxon signed-rank test if normality is violated (p < 0.05 on Shapiro-Wilk)
- **Null hypothesis**: Mean difference = 0 (systems perform equally)
- **Primary metric**: nDCG@10

## How It Works

1. For each query present in both systems, compute per-query nDCG@10 difference
2. Run Shapiro-Wilk test on the difference distribution
3. If normal (p ≥ 0.05): use paired t-test
4. If non-normal: use Wilcoxon signed-rank test
5. Significant if p < 0.05 (two-sided)

## Historical

Previously used paired bootstrap (10,000 resamples, 95% CI) on MAP@10. Updated to align with Al-Joofi 2025 methodology: nDCG@10 as primary, paired t-test + Shapiro-Wilk for significance.

## Usage in SEG

Applied to nDCG@10 across all four core datasets for the main pairwise comparisons:
- BGE-base vs SciNCL: significant everywhere (p < 0.001)
- Adaptive RRF vs BGE-base: not significant anywhere (p ≥ 0.139)
- CE on ARF vs ARF: significantly degrades on SciFact (p = 0.015)
- SGAF B5/P3/P4 diagnostics additionally use paired bootstrap on per-query nDCG@10 because those ablations are reported as frozen candidate checks rather than the original cross-dataset t-test table.

Script: `scripts/run_significance_tests.py`

## Related

- [[nDCG@10]] — primary metric used for significance testing
- [[MAP@10]] — secondary metric also tested
- [[BGE]] — primary comparison target
- [[Al-Joofi-et-al]] — reference paper using same methodology
