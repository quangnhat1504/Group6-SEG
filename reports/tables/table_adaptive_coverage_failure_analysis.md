# Adaptive Coverage Failure Analysis

| Dataset | Method | Changed | Helped | Hurt | Missed BGE-base rescue | Captured BGE-base rescue | Mean delta |
|---|---|---:|---:|---:|---:|---:|---:|
| fiqa | Uncertainty-shift adaptive coverage | 142 | 61 | 39 | 147 | 61 | +0.0138 |
| fiqa | Source-shift adaptive coverage | 89 | 39 | 22 | 169 | 39 | +0.0082 |
| fiqa | Conservative-shift adaptive coverage | 67 | 30 | 17 | 178 | 30 | +0.0045 |
| fiqa | Fixed A3 coverage 5% | 32 | 18 | 6 | 190 | 18 | +0.0017 |
| nfcorpus | Source-shift adaptive coverage | 118 | 51 | 40 | 66 | 51 | +0.0074 |
| nfcorpus | Uncertainty-shift adaptive coverage | 58 | 27 | 21 | 90 | 27 | +0.0055 |
| nfcorpus | Conservative-shift adaptive coverage | 46 | 23 | 17 | 94 | 23 | +0.0049 |
| nfcorpus | Fixed A3 coverage 5% | 16 | 7 | 6 | 110 | 7 | +0.0025 |
| scidocs | Uncertainty-shift adaptive coverage | 211 | 87 | 62 | 277 | 87 | +0.0068 |
| scidocs | Source-shift adaptive coverage | 162 | 67 | 43 | 297 | 67 | +0.0058 |
| scidocs | Conservative-shift adaptive coverage | 110 | 42 | 35 | 322 | 42 | +0.0022 |
| scidocs | Fixed A3 coverage 5% | 50 | 16 | 16 | 348 | 16 | +0.0008 |
| scifact | Conservative-shift adaptive coverage | 18 | 5 | 4 | 30 | 5 | +0.0037 |
| scifact | Source-shift adaptive coverage | 25 | 7 | 6 | 28 | 7 | +0.0030 |
| scifact | Uncertainty-shift adaptive coverage | 25 | 7 | 6 | 28 | 7 | +0.0030 |
| scifact | Fixed A3 coverage 5% | 15 | 4 | 4 | 31 | 4 | +0.0028 |

## Interpretation

- `captured_bge_base_rescue` counts queries where BGE-base beats the specialist and the method improves over the specialist.
- `missed_bge_base_rescue` counts remaining BGE-base-positive opportunities not recovered by the method.
- A good adaptive coverage policy should increase captured rescues faster than hurt cases.
