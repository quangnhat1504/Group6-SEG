# SGAF Phase 8 Validation and Contribution Tables

This report is generated from existing artifacts only. It does not retune Frozen B5.

## Frozen Replication Gate

Decision rule for V3: keep Frozen B5 if SciFact delta vs BGE-small is at least `-0.005` and transfer delta vs BGE-base is at least `-0.002`.

| Row | Method | SciFact | Transfer Avg | SciFact delta | Transfer delta vs BGE-base | Filtered SciFact | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | BGE-small specialist | 0.8188 | 0.3011 | +0.0000 | -0.0239 | 0.8176 | reference |
| V1 | BGE-base generalist | 0.7376 | 0.3251 | -0.0812 | +0.0000 | 0.7409 | reference |
| V2 | Current adaptive SGAF | 0.8218 | 0.3098 | +0.0030 | -0.0152 | 0.8206 | reference |
| V3 | Frozen B5 mode-switch SGAF | 0.8218 | 0.3249 | +0.0030 | -0.0001 | 0.8206 | pass |

## Additive Contribution

| Step | Method | Avg | Transfer Avg | Transfer delta vs previous | Transfer delta vs BGE-base | SciFact delta | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | BGE-small specialist | 0.4305 | 0.3011 |  | -0.0239 | +0.0000 | specialist only |
| C1 | Fixed A3 rescue | 0.4325 | 0.3028 | +0.0017 | -0.0222 | +0.0028 | source-trained BGE-base rescue, fixed 5% coverage |
| C2 | Current adaptive SGAF | 0.4378 | 0.3098 | +0.0070 | -0.0152 | +0.0030 | uncertainty coverage raises rescue budget modestly |
| C3 | Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | +0.0151 | -0.0001 | +0.0030 | batch mode switch enables high BGE-base coverage only when shifted |

## Batch Shift Diagnostics

| Dataset | Shift score | Mode | Coverage | nDCG@10 | Delta vs BGE-base | Delta vs current |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | 1.0868 | specialist_safe | 0.0836 | 0.8218 | +0.0842 | +0.0000 |
| nfcorpus | 6.0246 | generalist_fallback | 0.8285 | 0.3692 | -0.0003 | +0.0133 |
| fiqa | 3.7930 | generalist_fallback | 1.0000 | 0.3909 | +0.0000 | +0.0136 |
| scidocs | 3.5845 | generalist_fallback | 1.0000 | 0.2147 | +0.0000 | +0.0185 |

## Interpretation

- Fixed A3 verifies that a source-trained BGE-base rescue ranker is robust but conservative.
- Current uncertainty coverage adds transfer improvement but still underuses BGE-base on shifted batches.
- Frozen B5 contributes the largest transfer jump by adding batch-level mode switching.
- The BGE-base comparison should remain caveated: Frozen B5 nearly recovers transfer while preserving SciFact, but does not universally outperform BGE-base.
