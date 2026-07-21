# Final SGAF Synthesis

## Current Candidate

**Frozen P3 Rank-Window Smoothing SGAF** is the strongest current experimental candidate.

It builds on **Frozen B5 Mode-Switch SGAF**, which remains the core specialist/generalist mode-switch mechanism.

Frozen components:

- BGE-small-final as the SciFact specialist.
- BGE-base as the generalist fallback model.
- BGE-base rescue ranker trained on SciFact `trainfit`, `C=0.1`.
- Batch source-shift controller:

`shift = abs(z(query_len)) + max(0,-z(bge_small_top)) + max(0,-z(bge_small_gap)) + max(0,-z(bge_small_std10)) + max(0,-z(bge_small_bge_base_overlap10))`

Frozen recipe:

- if `shift < 2.0`: keep current uncertainty coverage;
- if `shift >= 2.0`: use shifted uncertainty gain `6.0` with cap `1.0`;
- shifted uncertainty coverage keeps the same base formula but scales the surplus term.

P3 post-retrieval smoothing recipe:

- apply only when Frozen B5 mode is `generalist_fallback`;
- rerank the top `20` window with a weak BGE-small specialist prior;
- use `alpha=0.10`, `rrf_k=60`;
- leave source-like SciFact unchanged.

Important caveat: B5 threshold `2.0` and gain `6.0` were selected after the Phase 7 B4/B5 exploration. P3 `window=20`, `alpha=0.10` was selected after Phase 8 smoothing exploration and supported by leave-one-transfer-dataset-out validation. Both are frozen now; any future validation must keep them fixed.

## Phase Summary

| Phase | Result | Evidence |
|---|---|---|
| Cheap BM25 repair | Rejected as main path | Clean trainfit->dev S9 delta +0.0005, below +0.005 criterion; BM25 rescue has no clean trainfit/dev signal. |
| Static ensemble | Rejected | Best dev weighted RRF delta only +0.00004 and assigns BGE-base weight 0. |
| Query-adaptive SGAF | Promising but small | A3 fixed 5% coverage improves dev +0.0046 and test +0.0028; not significant on SciFact. |
| Frozen transfer | Robust but conservative | Fixed A3 is directionally positive on all four datasets but underuses BGE-base on transfer. |
| Adaptive coverage | Useful predecessor | Uncertainty-shift improves over fixed A3, but still trails BGE-base transfer by -0.0152 average. |
| BGE-base mode switch | Core candidate | Frozen B5 preserves SciFact and nearly recovers BGE-base transfer average. |
| Rank-window smoothing | Current strongest experimental candidate | Frozen P3 preserves SciFact and raises transfer average above BGE-base on current evaluated datasets. |
| Residual specialist fallback | Optional extension | P4 adds `+0.0021` transfer over P3 in LOTO, but evidence is weaker than P3 and should not replace the main claim without external validation. |

## Final Candidate Metrics

| Method | Avg nDCG@10 | Transfer Avg | Avg delta vs BGE-small | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|---:|---:|
| BGE-small specialist | 0.4305 | 0.3011 | +0.0000 | -0.0239 | +0.0000 |
| BGE-base generalist | 0.4282 | 0.3251 | -0.0024 | +0.0000 | -0.0812 |
| Current adaptive SGAF | 0.4378 | 0.3098 | +0.0073 | -0.0152 | +0.0030 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | +0.0186 | -0.0001 | +0.0030 |
| Frozen P3 rank-window smoothing SGAF | 0.4524 | 0.3293 | +0.0218 | +0.0042 | +0.0030 |

Incremental contribution relative to BGE-base:

| Step | Added mechanism | Avg nDCG@10 | Transfer Avg | Delta vs previous transfer | Delta vs BGE-base transfer | SciFact nDCG@10 | Status |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | BGE-base generalist | 0.4282 | 0.3251 | - | +0.0000 | 0.7376 | baseline |
| 1 | B5 mode switch: specialist-safe vs generalist-fallback | 0.4492 | 0.3249 | -0.0001 | -0.0001 | 0.8218 | core mechanism |
| 2 | P3 rank-window smoothing inside generalist-fallback | 0.4524 | 0.3293 | +0.0043 | +0.0042 | 0.8218 | main candidate |
| 3 | P4 residual specialist fallback, top 10% confidence gap | 0.4540 | 0.3314 | +0.0021 | +0.0063 | 0.8218 | appendix diagnostic only |

Interpretation: B5 is valuable because it repairs the SciFact specialist/generalist conflict without giving up BGE-base transfer. P3 is the first promoted step that clears BGE-base transfer average. P4 adds a small positive residual gain, but current evidence is not strong enough to move it into the main method.

B5 contribution over current adaptive:

- Avg nDCG@10: `+0.0113`.
- Transfer Avg nDCG@10: `+0.0151`.
- SciFact delta: `+0.0000`, so it preserves the current adaptive SciFact result.

P3 contribution over Frozen B5:

- Avg nDCG@10: `+0.0032`.
- Transfer Avg nDCG@10: `+0.0043`.
- SciFact delta: `+0.0000`, because smoothing is disabled in `specialist_safe` mode.
- Leave-one-transfer-dataset-out selection chooses `window=20`, `alpha=0.10` in all three transfer folds.

P4 residual fallback diagnostic over Frozen P3:

- Recipe tested: in `generalist_fallback` batches, select the top `10%` queries by `BGE-small top score - BGE-base top score` and replace those queries with the BGE-small specialist run.
- Transfer Avg nDCG@10: `0.3314`, or `+0.0021` over P3 and `+0.0063` over BGE-base.
- LOTO held-out deltas vs P3 are all positive: NFCorpus `+0.0023`, FiQA `+0.0016`, SciDocs `+0.0024`.
- Bootstrap vs P3 is significant only on SciDocs; keep P4 as optional residual extension, not the main paper claim.

## Dataset Detail

| Dataset | Current adaptive | BGE-base | Frozen B5 | B5 delta vs current | B5 delta vs BGE-base | B5 mode |
|---|---:|---:|---:|---:|---:|---|
| SciFact | 0.8218 | 0.7376 | 0.8218 | +0.0000 | +0.0842 | specialist-safe |
| NFCorpus | 0.3559 | 0.3695 | 0.3692 | +0.0133 | -0.0003 | generalist-fallback |
| FiQA | 0.3774 | 0.3909 | 0.3909 | +0.0136 | +0.0000 | generalist-fallback |
| SciDocs | 0.1962 | 0.2147 | 0.2147 | +0.0185 | +0.0000 | generalist-fallback |

P3 detail:

| Dataset | Frozen B5 | Frozen P3 | P3 delta vs B5 | P3 delta vs BGE-base |
|---|---:|---:|---:|---:|
| SciFact | 0.8218 | 0.8218 | +0.0000 | +0.0842 |
| NFCorpus | 0.3692 | 0.3744 | +0.0052 | +0.0048 |
| FiQA | 0.3909 | 0.3960 | +0.0051 | +0.0051 |
| SciDocs | 0.2147 | 0.2173 | +0.0027 | +0.0027 |

## Duplicate-Filtered SciFact

| Method | Full nDCG@10 | Filtered nDCG@10 | Filtered delta vs BGE-small | Filtered delta vs current |
|---|---:|---:|---:|---:|
| BGE-small specialist | 0.8188 | 0.8176 | +0.0000 | -0.0030 |
| BGE-base generalist | 0.7376 | 0.7409 | -0.0767 | -0.0797 |
| Current adaptive SGAF | 0.8218 | 0.8206 | +0.0030 | +0.0000 |
| Frozen B5 mode-switch SGAF | 0.8218 | 0.8206 | +0.0030 | +0.0000 |
| Frozen P3 rank-window smoothing SGAF | 0.8218 | 0.8206 | +0.0030 | +0.0000 |

## Significance

10k paired bootstrap for Frozen B5:

| Dataset | Baseline | Mean delta | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---|
| SciFact | Current adaptive SGAF | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| SciFact | BGE-small specialist | +0.002955 | [-0.005719, +0.011935] | 0.5114 | no |
| SciFact | BGE-base generalist | +0.084150 | [+0.055194, +0.114215] | 0.0000 | yes |
| NFCorpus | Current adaptive SGAF | +0.013267 | [-0.000088, +0.027252] | 0.0510 | no |
| NFCorpus | BGE-small specialist | +0.018731 | [+0.003866, +0.034561] | 0.0134 | yes |
| NFCorpus | BGE-base generalist | -0.000323 | [-0.005133, +0.004672] | 0.8784 | no |
| FiQA | Current adaptive SGAF | +0.013555 | [-0.001059, +0.027949] | 0.0692 | no |
| FiQA | BGE-small specialist | +0.027398 | [+0.010672, +0.043733] | 0.0014 | yes |
| FiQA | BGE-base generalist | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |
| SciDocs | Current adaptive SGAF | +0.018481 | [+0.010483, +0.026447] | 0.0000 | yes |
| SciDocs | BGE-small specialist | +0.025329 | [+0.016543, +0.034359] | 0.0000 | yes |
| SciDocs | BGE-base generalist | +0.000000 | [+0.000000, +0.000000] | 1.0000 | no |

10k paired bootstrap for P3 LOTO held-out folds:

| Held-out | Baseline | Mean delta | 95% CI | p-value | Significant |
|---|---|---:|---:|---:|---|
| NFCorpus | Frozen B5 mode-switch SGAF | +0.005161 | [+0.000756, +0.009591] | 0.0218 | yes |
| NFCorpus | BGE-base generalist | +0.004838 | [-0.001601, +0.011563] | 0.1414 | no |
| FiQA | Frozen B5 mode-switch SGAF | +0.005093 | [-0.000891, +0.011005] | 0.0938 | no |
| FiQA | BGE-base generalist | +0.005093 | [-0.001022, +0.011171] | 0.1044 | no |
| SciDocs | Frozen B5 mode-switch SGAF | +0.002670 | [-0.000187, +0.005452] | 0.0670 | no |
| SciDocs | BGE-base generalist | +0.002670 | [-0.000149, +0.005437] | 0.0650 | no |

## Novelty

The contribution is not another dense retriever and not a static ensemble. It is a cheap two-level specialist-generalist retrieval policy:

1. A source-trained rescue ranker orders queries by likely BGE-base rescue value.
2. A batch-level source-shift controller decides whether the system should remain specialist-safe or allow high generalist coverage.
3. A cheap post-retrieval rank-window smoother reintroduces a weak specialist prior only inside shifted batches.
4. An optional residual fallback can recover a small subset of high-confidence specialist queries after P3, but current evidence is not strong enough to promote it above P3.

This makes the fallback logic explicit:

- source-like batch -> preserve specialist-first behavior;
- shifted batch -> escalate to BGE-base coverage without CrossEncoder or LLM reranking.
- shifted top-rank window -> blend a small specialist prior after fallback, still without CrossEncoder or LLM reranking.

## Claim Audit

| Claim | Status | Evidence |
|---|---|---|
| BGE-small is the strongest SciFact specialist among the compared dense components | Supported | BGE-small SciFact 0.8188 vs BGE-base 0.7376. |
| Current adaptive SGAF improves over BGE-small/fixed A3 | Supported | Positive on all four datasets vs BGE-small and fixed A3. |
| Current adaptive SGAF beats BGE-base as a universal retriever | Rejected | Transfer average is 0.3098 vs BGE-base 0.3251. |
| Cap-only increase solves the transfer gap | Rejected | Cap-only max 1.00 is identical to current uncertainty coverage. |
| Global uncertainty gain solves transfer without cost | Rejected | It recovers transfer but hurts SciFact. |
| Frozen B5 preserves SciFact and recovers BGE-base transfer | Supported on current evaluated datasets | SciFact unchanged at 0.8218; transfer average 0.3249 vs BGE-base 0.3251. |
| Frozen B5 is paper-grade without caveat | Not yet | Threshold/gain came from Phase 7 exploration; future validation must keep them fixed. |
| Frozen P3 improves over Frozen B5 on current transfer datasets | Supported with caveat | Transfer average 0.3293 vs B5 0.3249; LOTO selects the same variant in all folds. |
| Frozen P3 is paper-grade external validation | Not yet | The P3 grid was designed after seeing the current project datasets. |
| P4 residual fallback improves over Frozen P3 | Supported as diagnostic only | Transfer average 0.3314 vs P3 0.3293; LOTO positive on all transfer datasets, but bootstrap vs P3 is significant only on SciDocs. |

## Current Decision

- Promote Frozen P3 Rank-Window Smoothing SGAF as the strongest current experimental candidate.
- Keep Frozen B5 Mode-Switch SGAF as the core mechanism and paper-safer ablation.
- Keep uncertainty-shift adaptive coverage as the predecessor ablation.
- Keep P4 residual specialist fallback as an optional diagnostic extension only; do not promote it over P3 without external frozen validation.
- Future validation should keep B5 threshold/gain, P3 window/alpha, and any P4 fraction/feature frozen before evaluating on a new held-out batch or dataset.
