# SGAF Phase 8 Validation and Ablation Plan

Goal: turn Frozen B5 from a strong current candidate into a cleaner claim. The recipe must stay frozen unless a row is explicitly marked as an ablation.

Frozen B5 recipe:

- SciFact trainfit BGE-base rescue classifier: `C=0.1`
- Batch shift threshold: `2.0`
- Shifted uncertainty gain: `6.0`
- Shifted cap: `1.0`
- Current evaluated result: SciFact `0.8218`, transfer average `0.3249`, BGE-base transfer average `0.3251`

## Main Claim To Validate

Frozen B5 is not a universal replacement for BGE-base. The claim is:

> A cheap specialist-generalist mode switch preserves the SciFact specialist on source-like batches and recovers nearly all BGE-base transfer performance under distribution shift.

## Phase 8A: Frozen Replication Gate

Run this first before adding new logic.

Implemented command:

```powershell
python scripts\summarize_sgaf_phase8_validation.py
```

Generated artifacts:

- `runs/fusion/phase8_validation/phase8_replication_gate.csv`
- `runs/fusion/phase8_validation/phase8_contribution_ablation.csv`
- `runs/fusion/phase8_validation/phase8_shift_diagnostics.csv`
- `runs/fusion/phase8_validation/phase8_validation_manifest.json`
- `reports/tables/table_sgaf_phase8_validation_ablation.md`

| Row | Method | What changes | Expected evidence | Decision rule |
|---|---|---|---|---|
| V0 | BGE-small specialist | component baseline | specialist/source ceiling | reference only |
| V1 | BGE-base generalist | component baseline | transfer ceiling | reference only |
| V2 | Current adaptive SGAF | predecessor | transfer gap vs BGE-base | must remain below B5 |
| V3 | Frozen B5 | frozen recipe | source preservation + transfer recovery | keep if SciFact loss <= `0.005` and transfer delta vs BGE-base >= `-0.002` |

Primary metrics:

- nDCG@10 on each dataset
- transfer average over NFCorpus, FiQA, SciDocs
- SciFact duplicate-filtered nDCG@10
- paired bootstrap vs BGE-small, current adaptive, and BGE-base

## Phase 8B: Contribution Ablation

This table measures why B5 works. It should be reported as additive contribution, not only final score.

| Row | Additive component | Description | Hypothesis | Current or planned contribution signal |
|---|---|---|---|---|
| C0 | BGE-small specialist | no BGE-base rescue | strong SciFact, weak transfer | transfer avg `0.3011` |
| C1 | Fixed A3 rescue | source-trained BGE-base rescue ranker, fixed 5% coverage | robust but conservative | transfer avg improves to `0.3028` |
| C2 | Current uncertainty coverage | label-free per-query uncertainty raises coverage modestly | helps FiQA/SciDocs but remains below BGE-base | transfer avg `0.3098` |
| C3 | Batch shift detector | source-like vs shifted batch decision | separates SciFact from transfer datasets | shift scores: SciFact `1.087`, transfers `3.584-6.025` |
| C4 | Frozen B5 high-coverage mode | use gain `6.0`, cap `1.0` only when shifted | recover BGE-base transfer without SciFact loss | transfer avg `0.3249`, SciFact `0.8218` |

Recommended report table:

| Step | Method | SciFact | Transfer Avg | Delta vs previous | Delta vs BGE-base transfer | Interpretation |
|---|---|---:|---:|---:|---:|---|
| 0 | BGE-small | 0.8188 | 0.3011 | N/A | -0.0239 | specialist only |
| 1 | Fixed A3 | 0.8216 | 0.3028 | +0.0017 | -0.0222 | conservative rescue |
| 2 | Current adaptive | 0.8218 | 0.3098 | +0.0068 | -0.0152 | uncertainty helps but undercovers |
| 3 | Frozen B5 | 0.8218 | 0.3249 | +0.0151 | -0.0001 | mode switch recovers transfer |

Current result from generated artifact:

- Frozen B5 passes the gate.
- The largest measured contribution is the batch mode switch itself: transfer average moves from `0.3098` to `0.3249`, a `+0.0151` increment over current adaptive SGAF.
- Fixed A3 and current adaptive coverage remain important predecessor ablations, but neither explains the BGE-base transfer recovery by itself.

## Phase 8C: Cheap Post-Retrieval Improvements

Only run these after V3 is stable. These are cheap post-retrieval changes, not preprocessing and not LLM/CE reranking.

| Row | Method | Added cost | Logic | Risk | Keep if |
|---|---:|---:|---|---|---|
| P1 | score tie normalization | negligible | normalize final fused scores for display/threshold diagnostics only | should not alter ranking unless intentionally used | diagnostics become clearer |
| P2 | duplicate candidate collapse | negligible | if multiple chunks/doc variants map to same canonical doc, keep best rank before final top-k | can hurt if passage-level diversity matters | diagnostic-only on BEIR; maybe useful for web evidence diversity |
| P3 | rank-window smoothing | negligible | for B5 fallback batches, blend a small specialist prior into BGE-base ranks for a small top-rank window only | may reintroduce specialist noise on transfer | promising exploratory candidate; needs frozen validation |
| P4 | residual specialist fallback | cheap | only for shifted batches, detect high-confidence specialist residual queries and choose BGE-small for that query | can become B6 overfitting if tuned on target labels | optional extension; positive but weaker than P3 |

P2 current result:

- Implemented `scripts/run_sgaf_post_retrieval_collapse_ablation.py`.
- Generated `reports/tables/table_sgaf_phase8_post_retrieval_collapse.md`.
- Removed `1,927` repeated canonical hits across the evaluated rows, mostly NFCorpus.
- Frozen B5 delta is `-0.0001` mean nDCG@10 across datasets and `-0.0005` on NFCorpus, so this is **not** a performance contribution.
- Keep the idea only as a production-web evidence diversity cleanup, not as a paper claim.

P3 current result:

- Implemented `scripts/run_sgaf_rank_window_smoothing_ablation.py`.
- Implemented `scripts/summarize_sgaf_p3_loto_validation.py`.
- Implemented `scripts/summarize_sgaf_p3_final_candidate.py`.
- Generated `reports/tables/table_sgaf_phase8_rank_window_smoothing.md`.
- Generated `reports/tables/table_sgaf_p3_loto_validation.md`.
- Generated `reports/tables/table_final_sgaf_p3_smoothing.md`.
- Best exploratory row: `window=20`, `alpha=0.10`, `rrf_k=60`.
- Transfer average improves from Frozen B5 `0.3249` to `0.3293` (`+0.0043`) and exceeds BGE-base transfer by `+0.0042`.
- Dataset deltas vs Frozen B5:
  - NFCorpus `+0.0052`, significant by paired bootstrap (`p=0.0226`);
  - FiQA `+0.0051`, positive but not significant (`p=0.0958`);
  - SciDocs `+0.0027`, positive but not significant (`p=0.0678`);
  - SciFact unchanged because source-like batches are not smoothed.
- Leave-one-transfer-dataset-out selection also selects `window=20`, `alpha=0.10` in all three folds:
  - held-out NFCorpus `+0.0052` vs B5, significant (`p=0.0218`);
  - held-out FiQA `+0.0051` vs B5, positive but not significant (`p=0.0938`);
  - held-out SciDocs `+0.0027` vs B5, positive but not significant (`p=0.0670`).
- Decision: keep P3 as the next candidate after Frozen B5, but do not promote it to final claim until `window=20`, `alpha=0.10` is frozen and validated on a new split/batch.
- Frozen P3 candidate summary: avg nDCG@10 `0.4524`, transfer avg `0.3293`, transfer delta vs BGE-base `+0.0042`, transfer delta vs B5 `+0.0043`, SciFact delta vs BGE-small `+0.0030`.

Do not run expensive post-processing yet:

- CrossEncoder reranking is a negative ablation in the BGE pipeline.
- LLM reranking should be reserved as an upper-cost comparison, not the next optimization.

P4 residual fallback result:

- Implemented `scripts/run_sgaf_p4_residual_fallback_ablation.py`.
- Generated `reports/tables/table_sgaf_phase8_p4_residual_fallback.md`.
- Mechanism: after Frozen P3, only inside `generalist_fallback` batches, select the top fraction of queries by a label-free BGE-small confidence signal and replace those query rankings with BGE-small.
- Best sweep row: `p4_small_minus_base_top_frac_0_100`, where the feature is `BGE-small top score - BGE-base top score` and the fallback fraction is `10%`.
- Transfer average improves from Frozen P3 `0.3293` to `0.3314` (`+0.0021`) and exceeds BGE-base transfer by `+0.0063`.
- Leave-one-transfer-dataset-out selects the same variant in all three folds:
  - held-out NFCorpus `+0.0023` vs P3, not significant (`p=0.0610`);
  - held-out FiQA `+0.0016` vs P3, not significant (`p=0.5408`);
  - held-out SciDocs `+0.0024` vs P3, significant (`p=0.0422`).
- Decision: keep P4 as an optional residual extension. It is not strong enough to replace Frozen P3 as the main candidate because the gain is small and only one transfer dataset is significant vs P3.

## Phase 8D: Robustness Checks

| Check | Purpose | Failure means |
|---|---|---|
| Threshold sensitivity around frozen `2.0` | show B5 is not a razor-thin threshold artifact | need weaker claim or new validation split |
| Gain sensitivity around frozen `6.0` | show transfer recovery is not only one lucky gain | report as exploratory, not final |
| Leave-one-transfer-dataset-out reporting | show one dataset is not dominating transfer average | identify dataset-specific weakness |
| Duplicate-filtered SciFact | preserve source claim under leakage audit | cannot claim source preservation |
| New held-out batch/dataset without retuning | paper-grade validation | current result remains project-grade only |

## Stop Conditions

Do not add another residual fallback or expensive reranking if:

- Frozen B5 still has transfer delta vs BGE-base within `0.002`.
- SciFact remains unchanged after duplicate filtering.
- Any proposed post-retrieval tweak improves only one transfer dataset while hurting the others.

Promote a residual fallback only if:

- the residual gain remains consistent after frozen validation, and
- the fallback rule uses label-free features already available after retrieval, and
- an ablation shows contribution beyond Frozen P3 without reducing SciFact or transfer robustness.
