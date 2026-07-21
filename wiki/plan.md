---
title: "SEG Next Plan"
type: synthesis
tags: [roadmap, adaptive-fusion, cheap-repair, performance]
last_updated: 2026-07-20
---

# SEG Next Plan: Cheap Repair First, Then Specialist-Generalist Fusion

## Decision
The previous direction, **BEIR leak audit + multi-corpus labeled-only training**, is useful for credibility and ablations but is not the main novelty path.

The new execution order is:

1. Test **Specialist-First Cheap Retrieval Repair** first.
2. Only move to full **Specialist-Generalist Adaptive Fusion** if cheap repair cannot exploit the observed failure region.

This keeps the next step cost-aware: no query preprocessing, no Cross-Encoder, no LLM reranker, and no extra dense/generalist model before cheap BM25 repair is exhausted.

## Why
Current evidence shows a clear specialization/generalization tradeoff:

| Retriever | SciFact | NFCorpus | FiQA | SciDocs | Interpretation |
|-----------|---------|----------|------|---------|----------------|
| BGE-small final | **0.8188** | 0.3505 | 0.3635 | 0.1893 | strongest SciFact specialist |
| vstash rrf-v3 | 0.7707 | 0.3667 | **0.4825** | **0.2150** | stronger generalist/reference |
| BGE-base | 0.7376 | **0.3695** | 0.3909 | 0.2147 | safest simple baseline |

Training one more multi-corpus model may help, but the higher-value research question is: **when should the system trust a specialist, a generalist, or lexical BM25?**

## Current Cheap-Repair Finding

See [[Specialist-First Cheap Retrieval Repair]] and [[cheap-post-retrieval-repair-ablation]].

Phase 1 implementation is complete:

- Added `scripts/run_cheap_repair_ablation.py`.
- Ran dev/test multi-seed calibration/evaluation ablations.
- Created contribution tables in `runs/fusion/*_cheap_repair_multiseed_summary.csv`.

Key result:

- Test diagnostic has BM25 rescue headroom: oracle BM25 + BGE-small-final = `0.8624`, or `+0.0436` over BGE-small-final.
- Dev does not show the same BM25 rescue headroom; BGE-small-final dominates dev.
- Simple cheap rules are not robust enough yet:
  - dev S3 BM25 rerank mean delta: `+0.0001`;
  - dev S5 conditional BM25 injection mean delta: `-0.0004`;
  - dev S7 bounded BM25 promotion mean delta: `-0.0002`;
  - test diagnostic S3 mean delta: `-0.0017`;
  - test diagnostic S5 mean delta: `-0.0033`;
  - test diagnostic S7 bounded BM25 promotion mean delta: `-0.0007`.
- Failure analysis explains the tradeoff:
  - test S3 changes `0.955` of top-10 rankings, helps `14.0` queries/seed, hurts `12.2` queries/seed, and nets negative;
  - test S7 changes only `0.019` of top-10 rankings and limits hurt to `0.8` queries/seed, but misses `16.0` BM25 rescue opportunities/seed;
  - S5/S6 sit between those extremes and do not produce a robust gain.
- S8/S9 update:
  - dev S8 rank-local promotion mean delta: `-0.0007`;
  - dev S9 learned gate is no-op because dev calibration has no BM25 rescue-positive labels;
  - test diagnostic S8 mean delta: `+0.0005`;
  - test diagnostic S9 mean delta: `+0.0048`, range `[-0.0017, +0.0083]`, switch rate `0.5720`.
- Clean trainfit->dev validation:
  - generated `runs/scifact/trainfit_dense_bge_small_scifact_rrf.csv`;
  - trainfit specialist nDCG@10: `0.9332`;
  - trainfit BM25 rescue-positive queries: `2`;
  - dev BM25 rescue-positive queries: `0`;
  - clean S9 dev delta: `+0.0005`, below the `+0.005` success criterion.

Leader decision: cheap BM25 repair is exhausted as the main performance path. Keep S9 as a diagnostic/secondary baseline and move to Specialist-Generalist Adaptive Fusion or static ensemble validation.

## Protocol Guardrails
- Use SciFact `trainfit` for any learned gate or feature model.
- Use SciFact `dev` as the only iteration board.
- Freeze one recipe before final SciFact `test`.
- Report duplicate-filtered SciFact sensitivity because the source archive has two duplicate train/test claim-evidence pairs.
- Treat FiQA, NFCorpus, and SciDocs as zero-shot transfer evaluations unless explicit dev splits are created and frozen.
- Do not select model/fusion settings by cross-dataset test averages.

## Phase 1 — Cheap Post-Retrieval Repair
Goal: test whether BM25 can cheaply repair BGE-small-final without diluting the specialist.

Status: S0-S9 ablation and clean validation complete; not a robust win.

Completed tasks:
1. Created `scripts/run_cheap_repair_ablation.py`.
2. Evaluated:
   - BM25 baseline;
   - BGE-small-final baseline;
   - confidence diagnostics;
   - BM25 rerank inside specialist top-N;
   - always-on BM25 injection;
   - conditional BM25 injection;
   - conditional lexical BM25 rescue;
   - bounded BM25 promotion;
   - rank-local BM25 promotion;
   - learned cheap gate + rank-local promotion.
3. Ran false-positive/false-negative analysis and case-profile aggregation.
4. Generated trainfit BGE-small-final run.
5. Ran clean trainfit->dev validation.
6. Saved phase tables and wiki report.

Next tasks:
1. Treat cheap BM25 repair as a diagnostic section, not the final method.
2. Start Phase 2 static ensemble validation on dev.
3. If static ensemble fails, use its diagnostics to design query-adaptive specialist-generalist fusion.
4. Keep S9 as a secondary low-cost baseline in final tables.

Success criterion: dev improvement over BGE-small-final by at least `+0.005` nDCG@10 across seeds without lowering Recall@100 and without a high switch/promote rate.

## Phase 2 — Static Ensemble Baseline
Goal: find whether simple fusion already beats the specialist.

Status: complete; static ensemble does not produce a meaningful win.

Completed:
1. Created `scripts/run_finetuned_dense_retrieval.py`.
2. Generated `runs/scifact/dev_dense_bge_base.csv`.
3. Created `scripts/tune_static_ensemble.py`.
4. Swept weighted RRF over BM25, BGE-small-final, and BGE-base on SciFact dev.
5. Saved:
   - `runs/fusion/static_ensemble_dev/dev_static_ensemble_summary.csv`
   - `runs/fusion/static_ensemble_dev/dev_static_ensemble_best.json`
   - `reports/tables/table_static_ensemble_dev.md`

Result:

| Method | nDCG@10 | Delta vs BGE-small | Interpretation |
|---|---:|---:|---|
| BGE-small-final | 0.9052 | +0.0000 | specialist baseline |
| BGE-base | 0.7234 | -0.1818 | weaker global dev retriever |
| Best static weighted RRF | 0.9053 | +0.0000 | no meaningful gain; best gives BGE-base weight 0 |
| Oracle over BM25/BGE-small/BGE-base | 0.9235 | +0.0182 | query-adaptive headroom exists |

Decision: static global weighting is not the contribution. BGE-base rescues `10` dev queries, but global RRF dilutes the specialist. Move to query-adaptive fusion.

## Phase 3 — Query-Adaptive Fusion
Goal: make weights change per query.

Status: query-adaptive SGAF robustness checks complete; A3 is the current candidate but not a statistically significant win yet.

Completed:
1. Generated `runs/scifact/trainfit_dense_bge_base.csv`.
2. Created `scripts/train_query_adaptive_fusion.py`.
3. Trained/evaluated:
   - A1 multiclass component router;
   - A2 binary BGE-base rescue gate;
   - A3 coverage-controlled BGE-base rescue gate;
   - A4 coverage-controlled BM25 lexical rescue gate;
   - A5 dual BGE-base + BM25 rescue gate.
4. Saved:
   - `runs/fusion/query_adaptive_dev/trainfit_to_dev_query_adaptive_fusion.csv`
   - `runs/fusion/query_adaptive_test/trainfit_to_test_query_adaptive_fusion.csv`
   - `runs/fusion/query_adaptive_dev/dev_a3_significance.md`
   - `runs/fusion/query_adaptive_test/test_a3_significance.md`
   - `reports/tables/table_query_adaptive_fusion.md`

Result:

| Stage | Method | Dev Delta | Test Delta | Decision |
|---|---|---:|---:|---|
| A1 | Multiclass router | -0.0397 | -0.0361 | reject; over-switches |
| A2 | Absolute-threshold BGE-base rescue | +0.0065 | -0.0173 | reject; probability threshold shifts |
| A3 | Coverage-controlled BGE-base rescue | +0.0046 | +0.0028 | keep as current candidate |
| A4 | Coverage-controlled BM25 rescue | +0.0000 | +0.0000 | no-op; trainfit has no BM25 oracle-positive signal |
| A5 | Dual A3 + BM25 rescue | +0.0046 | +0.0028 | identical to A3; BM25 coverage selected as 0 |

Significance:

| Split | Mean delta | 95% CI | p-value | Significant |
|---|---:|---:|---:|---|
| dev | +0.004629 | [-0.003003, +0.013754] | 0.2530 | no |
| test | +0.002813 | [-0.003648, +0.010259] | 0.4304 | no |

Success status: A3 improves both dev and test with low switch rate, but dev gain is slightly below the earlier `+0.005` threshold and paired bootstrap is not significant. Treat as a promising mechanism/negative-ablation story, not a final performance claim.

Next tasks completed in Phase 4:
1. Evaluated cross-dataset transfer after freezing the SciFact A3 recipe.
2. Ran duplicate-filtered SciFact sensitivity.
3. Confirmed that the next useful direction is coverage adaptation, not more BM25 rescue.

## Phase 4 — Frozen Test
Goal: evaluate once after recipe selection.

Status: complete for frozen A3 SGAF.

Completed:
1. Added `scripts/evaluate_frozen_sgaf_transfer.py`.
2. Added `scripts/summarize_frozen_sgaf_robustness.py`.
3. Ran duplicate-filtered SciFact audit for BGE-small-final and frozen A3.
4. Ran zero-shot frozen A3 transfer on SciFact, NFCorpus, FiQA, and SciDocs.
5. Saved:
   - `runs/fusion/frozen_sgaf_transfer/frozen_sgaf_transfer_summary.csv`
   - `runs/fusion/frozen_sgaf_transfer/frozen_sgaf_transfer_significance.csv`
   - `reports/tables/table_frozen_sgaf_transfer.md`
   - `reports/tables/table_frozen_sgaf_robustness.md`

Duplicate-filtered SciFact:

| Method | Full nDCG@10 | Filtered nDCG@10 | Filtered delta vs BGE-small | Filtered Recall@10 |
|---|---:|---:|---:|---:|
| BGE-small-final | 0.8188 | 0.8176 | +0.0000 | 0.9345 |
| Frozen A3 SGAF | 0.8216 | 0.8204 | +0.0028 | 0.9378 |

Cross-dataset transfer:

| Dataset | BGE-small | BGE-base | Oracle | Frozen A3 | A3 delta | A3 significant |
|---|---:|---:|---:|---:|---:|---|
| SciFact | 0.8188 | 0.7376 | 0.8786 | 0.8216 | +0.0028 | no |
| NFCorpus | 0.3505 | 0.3695 | 0.4249 | 0.3530 | +0.0025 | yes, p=0.0488 |
| FiQA | 0.3635 | 0.3909 | 0.4650 | 0.3653 | +0.0017 | no |
| SciDocs | 0.1893 | 0.2147 | 0.2666 | 0.1902 | +0.0008 | no |

Success status: frozen A3 preserves specialist performance and is directionally positive on all four datasets, with one significant transfer gain on NFCorpus. The gains are still small because the frozen 5% coverage cap is conservative, especially where BGE-base is globally stronger than BGE-small.

Decision: SGAF novelty is real enough as a mechanism, but the next performance contribution should be **adaptive coverage control**: keep the specialist-first safety of A3, while letting coverage increase on dataset/query regions where generalist evidence is consistently stronger.

## Phase 5 — Adaptive Coverage Control
Goal: keep the specialist-first safety of A3, but let the BGE-base rescue budget grow under target distribution shift.

Status: exploratory adaptive coverage ablation complete.

Completed:
1. Added `scripts/run_adaptive_coverage_sgaf.py`.
2. Kept the BGE-base rescue ranking model frozen from SciFact `trainfit`.
3. Compared:
   - fixed A3 5% coverage;
   - source-shift adaptive coverage;
   - uncertainty-shift adaptive coverage;
   - conservative-shift adaptive coverage;
   - oracle coverage sweep as diagnostic upper bound.
4. Saved:
   - `runs/fusion/adaptive_coverage_sgaf/adaptive_coverage_sgaf_summary.csv`
   - `runs/fusion/adaptive_coverage_sgaf/adaptive_coverage_sgaf_significance.csv`
   - `runs/fusion/adaptive_coverage_sgaf/adaptive_coverage_failure_summary.csv`
   - `reports/tables/table_adaptive_coverage_sgaf.md`
   - `reports/tables/table_adaptive_coverage_failure_analysis.md`

Result:

| Dataset | Fixed A3 | Best adaptive in phase | Adaptive delta vs fixed A3 | Significant |
|---|---:|---:|---:|---|
| SciFact | 0.8216 | 0.8225 | +0.0009 | no |
| NFCorpus | 0.3530 | 0.3578 | +0.0049 | no |
| FiQA | 0.3653 | 0.3774 | +0.0121 | yes, p=0.0010 |
| SciDocs | 0.1902 | 0.1962 | +0.0060 | yes, p=0.0004 |

Interpretation:

- Adaptive coverage is the first post-A3 change with meaningful transfer gains while keeping SciFact stable.
- The BGE-base rescue ranking model was not retrained; the contribution comes from coverage control.
- The result is still exploratory because the coverage formulas were developed during this phase. Treat it as a strong next-candidate direction, not a final frozen claim.
- BM25 remains disabled: the clean protocol still provides no BM25-positive training signal.

Failure analysis:

| Dataset | Fixed captured rescue | Source-shift captured rescue | Uncertainty-shift captured rescue | BGE-base rescue headroom |
|---|---:|---:|---:|---:|
| SciFact | 4 | 7 | 7 | 35 |
| NFCorpus | 7 | 51 | 27 | 117 |
| FiQA | 18 | 39 | 61 | 208 |
| SciDocs | 16 | 67 | 87 | 364 |

Freeze recommendation:

- Freeze **uncertainty-shift coverage** if optimizing average transfer performance: it gives the strongest FiQA/SciDocs gains and the best 4-dataset average, but it also switches more and hurts more queries.
- Keep **source-shift coverage** as the safer alternative: it is best on NFCorpus and has a clearer domain-shift story.
- Do not use oracle coverage sweep except as headroom evidence.

Next tasks completed in Phase 6:
1. Freeze one adaptive coverage formula for the final candidate.
2. Record duplicate-filtered SciFact evidence for the frozen formula.
3. Create an overall phase/novelty synthesis.

## Phase 6 — Adaptive Coverage Synthesis
Goal: report the full ablation story, overall result, and novelty.

Status: complete; superseded by Phase 7 BGE-base mode switch.

Previous candidate: **Uncertainty-shift Adaptive Coverage SGAF**.

Formula:

`coverage = clamp(0.05 + 0.08*max(0,-z(bge_small_gap)) + 0.04*max(0,-z(bge_small_std10)), 0.02, 0.40)`

Saved:

- `runs/fusion/final_sgaf/final_sgaf_candidate_summary.csv`
- `runs/fusion/final_sgaf/final_sgaf_manifest.json`
- `reports/tables/table_final_sgaf_synthesis.md` (later updated to the Frozen B5 final synthesis)

Previous candidate metrics:

| Dataset | BGE-small | Fixed A3 | Final adaptive | Delta vs BGE-small | Delta vs Fixed A3 | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.8188 | 0.8216 | 0.8218 | +0.0030 | +0.0001 | 0.084 |
| NFCorpus | 0.3505 | 0.3530 | 0.3559 | +0.0055 | +0.0030 | 0.180 |
| FiQA | 0.3635 | 0.3653 | 0.3774 | +0.0138 | +0.0121 | 0.219 |
| SciDocs | 0.1893 | 0.1902 | 0.1962 | +0.0068 | +0.0060 | 0.211 |

Overall:

- Average delta vs BGE-small: `+0.0073`.
- Average delta vs fixed A3: `+0.0053`.
- Positive vs BGE-small on `4/4` datasets.
- Positive vs fixed A3 on `4/4` datasets.
- Significant vs fixed A3 on FiQA and SciDocs.

Novelty:

> Adaptive Coverage SGAF separates query ranking from budget selection: a source-trained specialist-generalist rescue ranker orders queries by likely generalist rescue value, while a cheap uncertainty-shift controller adjusts how much generalist coverage is allowed under domain shift.

Completion caveat:

- The adaptive coverage formula was selected after Phase 5 exploratory ablation. It is now a predecessor ablation rather than the final candidate; B5/P3 supersede it, and any paper-grade claim still needs frozen held-out validation.
- Oracle headroom remains large, so failure-driven mining is a future extension, not required for the current ablation goal.

## Secondary Baselines
- Multi-corpus labeled-only training: keep as an ablation, not the core novelty claim.
- Score-normalized CombSUM/CombMNZ: compare against weighted RRF.
- Candidate union before reranking: only revisit if fusion leaves clear rank-order failures.

## Main Claim Target
> Fine-tuning creates a strong SciFact specialist but weakens transfer. Adaptive Coverage SGAF preserves the specialist-first default while increasing generalist rescue coverage under cheap uncertainty-shift signals, improving transfer without retraining retrieval models.

## Phase 7 — BGE-base Mode-Switch Plan
Goal: close the transfer gap to BGE-base without giving up SciFact specialist performance.

Status: planned.

Evidence update:

| Dataset | BGE-base | Current adaptive | Adaptive - BGE-base | Current coverage | Oracle coverage |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.7376 | 0.8218 | +0.0842 | 0.084 | 0.050 |
| NFCorpus | 0.3695 | 0.3559 | -0.0136 | 0.180 | 1.000 |
| FiQA | 0.3909 | 0.3774 | -0.0136 | 0.219 | 1.000 |
| SciDocs | 0.2147 | 0.1962 | -0.0185 | 0.211 | 1.000 |

Interpretation:

- Current adaptive SGAF is better than BGE-small/fixed A3, but BGE-base remains stronger on transfer-only average.
- The main blocker is the coverage policy, not the existence of a generalist model.
- The next controller should add a generalist-fallback mode for high-shift cases while keeping specialist-safe low coverage for source-like cases.

Saved plan:

- `reports/sgaf_bge_base_mode_switch_plan.md`

Next execution order:

1. Implement B4 cap sweep: current uncertainty formula with max caps `0.40`, `0.60`, `0.80`, `1.00`. Completed.
2. Since cap-only is a no-op, implement batch-level source-shift mode rather than a global gain increase.
3. Add per-query fallback only after batch-level mode is measured.
4. Report contribution as an additive ablation table, not as a single final number.

B4 result:

- Added `scripts/run_sgaf_mode_switch_ablation.py`.
- Outputs:
  - `runs/fusion/sgaf_mode_switch_ablation/sgaf_mode_switch_ablation_rows.csv`
  - `runs/fusion/sgaf_mode_switch_ablation/sgaf_mode_switch_ablation_summary.csv`
  - `runs/fusion/sgaf_mode_switch_ablation/sgaf_mode_switch_ablation_significance.csv`
  - `reports/tables/table_sgaf_mode_switch_ablation.md`

| Method | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|
| Current uncertainty coverage | 0.3098 | -0.0152 | +0.0030 |
| Cap-only max 1.00 | 0.3098 | -0.0152 | +0.0030 |
| Gain 2.0 max 1.00 | 0.3128 | -0.0122 | -0.0051 |
| Gain 4.0 max 1.00 | 0.3176 | -0.0074 | -0.0097 |
| Gain 8.0 max 1.00 | 0.3251 | +0.0000 | -0.0103 |

Decision after B4:

- The hard cap is not the immediate blocker; cap-only rows are identical to current uncertainty coverage.
- The current formula is too conservative before it reaches the cap.
- Global gain can recover transfer, but it hurts SciFact, so the next implementation should be B5 batch-level source-shift mode, not a global gain increase.

B5 result:

- Added batch-level source-shift mode inside `scripts/run_sgaf_mode_switch_ablation.py`.
- Shift score cleanly separates current evaluated SciFact from transfer batches:
  - SciFact `1.087`;
  - NFCorpus `6.025`;
  - FiQA `3.793`;
  - SciDocs `3.584`.

| Method | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|
| Current uncertainty coverage | 0.3098 | -0.0152 | +0.0030 |
| Batch shift t2.0 gain 4.0 | 0.3176 | -0.0074 | +0.0030 |
| Batch shift t2.0 gain 6.0 | 0.3249 | -0.0001 | +0.0030 |
| Batch shift t2.0 gain 8.0 | 0.3251 | +0.0000 | +0.0030 |

Decision after B5:

- Mode switch is validated as a stronger path than global gain.
- `Batch shift t2.0 gain 6.0` is the best conservative next candidate: it nearly matches BGE-base transfer while preserving current SciFact adaptive score.
- `Batch shift t2.0 gain 8.0` fully matches BGE-base transfer but is more aggressive because it routes several transfer batches to pure BGE-base.
- Next at that point: freeze one B5 recipe and run duplicate-filtered SciFact plus bootstrap comparisons before trying any residual per-query fallback.

Frozen B5 result:

- Added `scripts/summarize_sgaf_mode_switch_final.py`.
- Frozen recipe: threshold `2.0`, shifted gain `6.0`, shifted cap `1.0`, SciFact trainfit BGE-base rescue classifier `C=0.1`.
- Outputs:
  - `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_summary.csv`
  - `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_duplicate_filtered_scifact.csv`
  - `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_significance.csv`
  - `reports/tables/table_final_sgaf_mode_switch.md`

| Method | Avg nDCG@10 | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|---:|
| BGE-small specialist | 0.4305 | 0.3011 | -0.0239 | +0.0000 |
| BGE-base generalist | 0.4282 | 0.3251 | +0.0000 | -0.0812 |
| Current adaptive SGAF | 0.4378 | 0.3098 | -0.0152 | +0.0030 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | -0.0001 | +0.0030 |

Duplicate-filtered SciFact stays stable:

- Current adaptive SGAF filtered nDCG@10: `0.8206`.
- Frozen B5 filtered nDCG@10: `0.8206`.

Decision after frozen B5:

- At this stage, B5 was the strongest candidate.
- Per-query fallback was not the immediate next step because B5 already recovered BGE-base transfer average while preserving SciFact. This was later superseded by the P3 smoothing and P4 residual diagnostics below.
- The next risk is claim cleanliness: threshold/gain came from exploratory B4/B5 sweep, so future validation must keep them frozen.

Claim audit:

- Added `reports/tables/table_sgaf_claim_audit.md`.
- At that stage, `reports/tables/table_final_sgaf_synthesis.md` promoted Frozen B5 Mode-Switch SGAF over uncertainty-shift adaptive coverage. This was later superseded by Frozen P3 as the strongest current experimental candidate.
- Supported claim: Frozen B5 preserves SciFact and nearly recovers BGE-base transfer on the current evaluated datasets.
- Rejected claim: the method universally beats BGE-base.
- Caveat: threshold/gain must remain frozen for any future validation.

## Phase 8: Frozen B5 Validation and Cheap Post-Retrieval Ablations

Plan artifact:

- `reports/sgaf_phase8_validation_ablation_plan.md`
- `scripts/summarize_sgaf_phase8_validation.py`
- `reports/tables/table_sgaf_phase8_validation_ablation.md`

Execution order:

1. Re-run the frozen replication gate first: BGE-small, BGE-base, current adaptive SGAF, and Frozen B5 with the exact frozen recipe. Completed from existing artifacts.
2. Report contribution as an additive table: specialist only -> fixed A3 -> current adaptive coverage -> Frozen B5 mode switch. Completed.
3. Keep threshold `2.0`, gain `6.0`, cap `1.0`, and classifier `C=0.1` fixed for any new validation batch.
4. Only after the frozen gate is stable, test cheap post-retrieval improvements:
   - duplicate/canonical candidate collapse; completed as diagnostic-only.
   - rank-window smoothing for shifted batches; completed as exploratory candidate.
   - residual per-query fallback inside shifted batches only.
5. Do not add CrossEncoder or LLM reranking yet; those are expensive upper-cost comparisons, not the next cheap optimization path.

Phase 8A gate result:

| Row | Method | SciFact | Transfer Avg | Transfer delta vs BGE-base | Gate |
|---|---|---:|---:|---:|---|
| V0 | BGE-small specialist | 0.8188 | 0.3011 | -0.0239 | reference |
| V1 | BGE-base generalist | 0.7376 | 0.3251 | +0.0000 | reference |
| V2 | Current adaptive SGAF | 0.8218 | 0.3098 | -0.0152 | reference |
| V3 | Frozen B5 mode-switch SGAF | 0.8218 | 0.3249 | -0.0001 | pass |

Contribution result:

| Step | Method | Transfer Avg | Delta vs previous | Interpretation |
|---|---|---:|---:|---|
| C0 | BGE-small specialist | 0.3011 | N/A | specialist only |
| C1 | Fixed A3 rescue | 0.3028 | +0.0017 | source-trained rescue is robust but conservative |
| C2 | Current adaptive SGAF | 0.3098 | +0.0070 | uncertainty coverage helps but undercovers transfer |
| C3 | Frozen B5 mode-switch SGAF | 0.3249 | +0.0151 | batch mode switch is the main contribution |

Phase 8C P2 result:

- Added `scripts/run_sgaf_post_retrieval_collapse_ablation.py`.
- Outputs:
  - `runs/fusion/phase8_post_retrieval/phase8_post_retrieval_collapse_rows.csv`
  - `runs/fusion/phase8_post_retrieval/phase8_post_retrieval_collapse_summary.csv`
  - `runs/fusion/phase8_post_retrieval/phase8_post_retrieval_collapse_manifest.json`
  - `reports/tables/table_sgaf_phase8_post_retrieval_collapse.md`
- It removes repeated canonical documents within a query ranking, keeping the first occurrence.
- Result: `1,927` repeated canonical hits removed across evaluated rows, but performance effect is negligible.
- Frozen B5 mean delta nDCG@10 across datasets is `-0.0001`; NFCorpus delta is `-0.0005`.
- Decision: do not count duplicate/canonical collapse as a B5 performance contribution. Keep it only as a possible web-facing evidence diversity cleanup.

Phase 8C P3 result:

- Added `scripts/run_sgaf_rank_window_smoothing_ablation.py`.
- Added `scripts/summarize_sgaf_p3_loto_validation.py`.
- Added `scripts/summarize_sgaf_p3_final_candidate.py`.
- Outputs:
  - `runs/fusion/phase8_rank_window_smoothing/phase8_rank_window_smoothing_rows.csv`
  - `runs/fusion/phase8_rank_window_smoothing/phase8_rank_window_smoothing_summary.csv`
  - `runs/fusion/phase8_rank_window_smoothing/phase8_rank_window_smoothing_significance.csv`
  - `runs/fusion/phase8_rank_window_smoothing/phase8_rank_window_smoothing_manifest.json`
  - `runs/fusion/phase8_rank_window_smoothing/phase8_p3_loto_rows.csv`
  - `runs/fusion/phase8_rank_window_smoothing/phase8_p3_loto_significance.csv`
  - `runs/fusion/phase8_rank_window_smoothing/phase8_p3_loto_manifest.json`
  - `reports/tables/table_sgaf_phase8_rank_window_smoothing.md`
  - `reports/tables/table_sgaf_p3_loto_validation.md`
  - `runs/fusion/final_sgaf_p3_smoothing/final_sgaf_p3_smoothing_rows.csv`
  - `runs/fusion/final_sgaf_p3_smoothing/final_sgaf_p3_smoothing_summary.csv`
  - `runs/fusion/final_sgaf_p3_smoothing/final_sgaf_p3_smoothing_duplicate_filtered_scifact.csv`
  - `runs/fusion/final_sgaf_p3_smoothing/final_sgaf_p3_smoothing_manifest.json`
  - `reports/tables/table_final_sgaf_p3_smoothing.md`
- Logic: only when Frozen B5 has already selected `generalist_fallback`, reorder a small top-rank window by blending B5 ranks with a weak BGE-small specialist prior. SciFact is left unchanged.
- Best exploratory row: `window=20`, `alpha=0.10`, `rrf_k=60`.

| Dataset | B5 | P3 best | Delta vs B5 | Significant vs B5 |
|---|---:|---:|---:|---|
| SciFact | 0.8218 | 0.8218 | +0.0000 | no-op |
| NFCorpus | 0.3692 | 0.3744 | +0.0052 | yes, `p=0.0226` |
| FiQA | 0.3909 | 0.3960 | +0.0051 | no, `p=0.0958` |
| SciDocs | 0.2147 | 0.2173 | +0.0027 | no, `p=0.0678` |

Aggregate:

- Transfer avg: `0.3249 -> 0.3293`, delta `+0.0043` vs Frozen B5.
- Transfer delta vs BGE-base: `+0.0042`.
- Leave-one-transfer-dataset-out validation:

| Held-out | Selected on | Selected variant | Held-out delta vs B5 | Significant |
|---|---|---|---:|---|
| NFCorpus | FiQA, SciDocs | `window=20, alpha=0.10` | +0.0052 | yes, `p=0.0218` |
| FiQA | NFCorpus, SciDocs | `window=20, alpha=0.10` | +0.0051 | no, `p=0.0938` |
| SciDocs | NFCorpus, FiQA | `window=20, alpha=0.10` | +0.0027 | no, `p=0.0670` |

- Decision: P3 is the best current post-retrieval candidate and LOTO reduces the overfit concern because every fold selects the same variant and every held-out delta is positive. It is still not paper-grade external validation because the grid was designed after seeing these project datasets. Freeze `window=20`, `alpha=0.10` before any future validation.

Frozen P3 candidate summary:

| Method | Avg nDCG@10 | Transfer Avg | Transfer delta vs BGE-base | Transfer delta vs B5 | SciFact delta |
|---|---:|---:|---:|---:|---:|
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | -0.0001 | +0.0000 | +0.0030 |
| Frozen P3 rank-window smoothing SGAF | 0.4524 | 0.3293 | +0.0042 | +0.0043 | +0.0030 |

P4 residual fallback diagnostic:

- Added `scripts/run_sgaf_p4_residual_fallback_ablation.py`.
- Report: `reports/tables/table_sgaf_phase8_p4_residual_fallback.md`.
- Logic: after P3, only inside `generalist_fallback` batches, select high-confidence specialist-residual queries and replace those query rankings with BGE-small.
- Best LOTO-selected variant: `p4_small_minus_base_top_frac_0_100`, where the feature is `BGE-small top score - BGE-base top score` and the fallback budget is `10%`.

| Held-out | P3 | P4 | Delta vs P3 | Delta vs BGE-base | Significant vs P3 |
|---|---:|---:|---:|---:|---|
| NFCorpus | 0.3744 | 0.3767 | +0.0023 | +0.0071 | no, `p=0.0610` |
| FiQA | 0.3960 | 0.3977 | +0.0016 | +0.0067 | no, `p=0.5408` |
| SciDocs | 0.2173 | 0.2198 | +0.0024 | +0.0051 | yes, `p=0.0422` |

- Decision: P4 is positive but small. Keep it as an optional residual extension, not as the main claim, unless external frozen validation confirms the gain.

Current status:

- Frozen P3 is the strongest current experimental candidate.
- Frozen B5 remains the core mode-switch mechanism and the cleaner paper-safe ablation.
- Do not present P3 as a generally validated BGE-base replacement; per-dataset bootstrap vs BGE-base is positive but not significant on transfer datasets.

Decision rule:

- Use `reports/sgaf_frozen_external_validation_protocol.md` as the next-step guardrail.
- Use `reports/sgaf_external_validation_candidate_matrix.md` first, then `reports/sgaf_external_validation_runbook.md`, `configs/sgaf_external_candidate_acquisition.template.json`, `scripts/init_sgaf_external_candidate_acquisition.py`, `scripts/download_sgaf_external_beir_candidate.py`, `scripts/audit_sgaf_external_candidate_acquisition.py`, `configs/sgaf_external_validation_manifest.template.json`, `scripts/prepare_sgaf_external_beir_dataset.py`, `scripts/init_sgaf_external_validation_manifest.py`, `scripts/audit_sgaf_external_dataset_choice.py`, `scripts/audit_sgaf_external_validation_manifest.py`, `scripts/audit_sgaf_external_data_readiness.py`, `scripts/run_sgaf_external_baseline_retrieval.py`, `scripts/run_sgaf_external_frozen_runs.py`, `scripts/audit_sgaf_external_run_readiness.py`, `scripts/score_sgaf_external_validation.py`, and `scripts/audit_sgaf_external_validation_results.py` when a new held-out batch or dataset becomes available.
- Keep Frozen B5 if source loss stays within `0.005` and transfer delta vs BGE-base stays within `0.002`.
- Promote Frozen P3 only if the frozen `window=20`, `alpha=0.10` recipe remains positive over B5 without any dataset dropping below `-0.002`.
- Add P4 to the main method only if frozen external validation confirms residual gain beyond P3 without hurting transfer/source robustness; otherwise keep P4 appendix-only.
