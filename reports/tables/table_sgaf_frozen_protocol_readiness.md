# SGAF Frozen Protocol Readiness Audit

This audit checks whether the current project artifacts are ready for a future frozen external-validation run. It does not rerun retrieval and does not retune any SGAF parameter.

## Current Frozen Metrics

| Method | Transfer Avg | Delta vs BGE-base transfer | SciFact nDCG@10 |
|---|---:|---:|---:|
| BGE-base generalist | 0.3251 | +0.0000 | 0.7376 |
| Frozen B5 mode-switch SGAF | 0.3249 | -0.0001 | 0.8218 |
| Frozen P3 rank-window smoothing SGAF | 0.3293 | +0.0042 | 0.8218 |
| Optional P4 residual fallback | 0.3314 | +0.0063 | 0.8218 |

## Gate Audit

| Gate | Status | Evidence | Action |
|---|---|---|---|
| Required artifacts present | pass | all required files found | continue |
| B5 frozen recipe | pass | threshold=2.0, gain=6.0, shifted_cap=1.0, c_value=0.1 | do not retune threshold/gain/cap/C |
| P3 frozen recipe | pass | window=20, alpha=0.1, rrf_k=60, apply_only_when_b5_mode=generalist_fallback | run unchanged on new batch |
| P4 optional recipe | pass | feature=small_minus_base_top, fraction=0.1, decision=candidate | evaluate appendix-only unless external promotion gate passes |
| Current source preservation gate | pass | duplicate-filtered SciFact delta vs BGE-small = +0.0030 | external SciFact-like validation should keep loss <= 0.005 |
| Current B5 transfer recovery gate | pass | B5 transfer delta vs BGE-base = -0.0001 | external validation must keep this >= -0.002 |
| Current P3 contribution gate | pass | transfer delta vs B5 = +0.0043, vs BGE-base = +0.0042 | external validation must show positive P3 contribution without dataset-level regressions |
| Current P4 promotion gate | fail | transfer delta vs P3 = +0.0021, min dataset delta = +0.0016, significant transfer datasets vs P3 = 1 | keep P4 appendix-only until an external validation batch passes the promotion gate |
| External validation availability | waiting | no new held-out batch/dataset artifact is registered in this audit | run frozen protocol on a new batch before strengthening claims |

## Decision

- The current artifact set is ready to be used as the frozen baseline package for a new held-out validation run.
- Frozen P3 remains the main candidate because it clears the current BGE-base transfer average and preserves SciFact.
- P4 remains appendix-only: current evidence has only one significant transfer dataset versus P3, below the promotion gate.
- No stronger paper claim should be made until a new external batch/dataset is evaluated with the frozen protocol.
