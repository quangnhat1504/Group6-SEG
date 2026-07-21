# SGAF Claim Audit

This audit separates what the current evidence supports from what should remain exploratory.

## Supported Claims

| Claim | Evidence |
|---|---|
| Fine-tuning created a strong SciFact specialist but hurt transfer. | BGE-small SciFact `0.8188`; BGE-base transfer average `0.3251` vs BGE-small `0.3011`. |
| Static global ensemble is not the contribution. | Best weighted RRF on dev gives only `+0.00004` and assigns BGE-base weight `0`. |
| Fixed A3 is robust but too conservative. | Fixed 5% coverage is positive vs BGE-small but remains far below BGE-base on transfer. |
| Current uncertainty adaptive coverage helps over fixed A3. | Significant gains over fixed A3 on FiQA and SciDocs. |
| Cap-only increase is not enough. | Cap-only max `1.00` is identical to current uncertainty coverage. |
| Global gain is the wrong final mechanism. | Gain `6.0/8.0` recovers transfer but drops SciFact below the preservation target. |
| Frozen B5 is the strongest mode-switch core candidate. | Avg nDCG `0.4492`; transfer avg `0.3249`; SciFact delta vs BGE-small `+0.0030`. |
| Frozen P3 is the strongest current experimental candidate. | Avg nDCG `0.4524`; transfer avg `0.3293`; SciFact unchanged vs B5. |
| P4 residual specialist fallback is a positive optional extension. | Transfer avg `0.3314`; LOTO deltas vs P3 are all positive, but only SciDocs is significant vs P3. |

## Rejected Claims

| Claim | Why rejected |
|---|---|
| Current adaptive SGAF beats BGE-base everywhere. | It loses to BGE-base on NFCorpus, FiQA, and SciDocs. |
| The hard max cap is the main transfer blocker. | Current coverage values are below the cap; raising only the cap changes nothing. |
| BM25 rescue should be restored as the main path. | Clean trainfit/dev selection gives no BM25-positive signal. |
| Per-query fallback is the main claim. | P4 is positive but small; B5/P3 remain the main mechanism and candidate. |

## Caveated Claims

| Claim | Caveat |
|---|---|
| Frozen B5 is the core mode-switch mechanism. | Supported as the clean ablation that preserves SciFact and nearly recovers BGE-base transfer, but threshold/gain were discovered during Phase 7 exploration. |
| Frozen P3 is the current strongest experimental candidate. | Supported on the current evaluated datasets, but window/alpha were discovered during Phase 8 exploration and still need external frozen validation. |
| Frozen B5 generalizes as a label-free controller. | The controller uses no target labels at application time, but the threshold/gain still need future frozen validation. |
| Frozen B5 matches BGE-base transfer. | It nearly matches transfer average and exactly matches FiQA/SciDocs, but NFCorpus remains `-0.0003`. |
| Frozen P3 improves over Frozen B5. | Supported on the current transfer datasets and LOTO folds, but the smoothing grid was designed after seeing the project datasets. |
| Frozen P3 beats BGE-base transfer. | Current transfer average is higher (`0.3293` vs `0.3251`), but per-dataset bootstrap vs BGE-base is not significant. |
| P4 improves over Frozen P3. | Directionally positive in LOTO (`+0.0021` transfer avg), but only one transfer dataset is significant vs P3 and the rule needs external frozen validation. |

## Recommended Paper Wording

Use:

> We propose a specialist-generalist mode-switching retrieval policy that preserves a fine-tuned SciFact specialist on source-like batches and escalates to BGE-base generalist coverage under cheap distribution-shift evidence.

Avoid:

> Our method universally outperforms BGE-base.

Use:

> On the current evaluated transfer datasets, frozen B5 nearly recovers BGE-base transfer performance while preserving the SciFact specialist gain.

Use:

> As an exploratory post-retrieval extension, frozen P3 rank-window smoothing improves over Frozen B5 on all current transfer datasets and keeps SciFact unchanged, but still requires external frozen validation.

Avoid:

> The threshold is fully validated as generally optimal.

Avoid:

> Frozen P3 is conclusively better than BGE-base in general.
