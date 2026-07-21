# SGAF Frozen External Validation Protocol

Purpose: evaluate the current SGAF recipe on the next held-out batch or dataset without retuning. This protocol is the guardrail that decides whether Frozen P3 can move from "strongest current experimental candidate" to a stronger paper claim, and whether optional P4 is worth promoting beyond appendix.

Operational runbook: `reports/sgaf_external_validation_runbook.md`.
Candidate matrix: `reports/sgaf_external_validation_candidate_matrix.md`.
Candidate acquisition template: `configs/sgaf_external_candidate_acquisition.template.json`.
Manifest template: `configs/sgaf_external_validation_manifest.template.json`.
Current package handoff: `reports/sgaf_package_handoff.md`.
Candidate acquisition initializer: `scripts/init_sgaf_external_candidate_acquisition.py`.
BEIR candidate downloader: `scripts/download_sgaf_external_beir_candidate.py`.
Candidate acquisition audit: `scripts/audit_sgaf_external_candidate_acquisition.py`.
Manifest initializer: `scripts/init_sgaf_external_validation_manifest.py`.
Dataset-choice audit: `scripts/audit_sgaf_external_dataset_choice.py`.
Manifest audit: `scripts/audit_sgaf_external_validation_manifest.py`.
Data-readiness audit: `scripts/audit_sgaf_external_data_readiness.py`.
Run-readiness audit: `scripts/audit_sgaf_external_run_readiness.py`.
Baseline retrieval runner: `scripts/run_sgaf_external_baseline_retrieval.py`.
Frozen SGAF run generator: `scripts/run_sgaf_external_frozen_runs.py`.
BEIR/local data prep: `scripts/prepare_sgaf_external_beir_dataset.py`.
Scorer: `scripts/score_sgaf_external_validation.py`.
Result-gate audit: `scripts/audit_sgaf_external_validation_results.py`.

## Frozen Recipes

These settings must not be changed during external validation:

| Component | Frozen value | Allowed during validation |
|---|---|---|
| Specialist retriever | BGE-small SciFact specialist | run as-is |
| Generalist retriever | BGE-base | run as-is |
| Rescue ranker | BGE-base rescue classifier, `C=0.1`, trained on SciFact `trainfit` | use existing model/features only |
| B5 mode switch | threshold `2.0`, shifted gain `6.0`, cap `1.0` | apply unchanged |
| P3 smoothing | only in `generalist_fallback`, `window=20`, `alpha=0.10`, `rrf_k=60` | apply unchanged |
| P4 optional fallback | only in `generalist_fallback`, feature `small_minus_base_top`, fraction `0.10` | evaluate as appendix candidate only |

No new threshold, feature, fraction, model, or dataset-specific weight may be selected from the external validation labels.

## Required Inputs

The new evaluation batch must provide:

| Input | Requirement |
|---|---|
| Corpus and queries | fixed before running SGAF |
| Relevance labels | hidden until all rankings are generated, or at least not used for recipe selection |
| Baseline runs | BM25, BGE-small specialist, BGE-base generalist |
| SGAF runs | current adaptive SGAF, Frozen B5, Frozen P3, optional P4 |
| Per-query records | ranking, nDCG@10, mode, shift score, selected/fallback flag |
| Cost records | online latency or abstract cost for each stage |

If the dataset has official dev/test splits, use dev only for sanity checks that do not change frozen parameters, and report test as the validation result.

## Evaluation Order

Before running on a new held-out batch, audit the current frozen package:

```powershell
python scripts\audit_sgaf_package_readiness.py
python scripts\audit_sgaf_external_candidate_acquisition.py --allow-template
python scripts\audit_sgaf_external_dataset_choice.py --allow-template
python scripts\audit_sgaf_external_validation_manifest.py --allow-template
python scripts\audit_sgaf_external_data_readiness.py --allow-template
python scripts\audit_sgaf_external_run_readiness.py --allow-template
python scripts\audit_sgaf_frozen_protocol_readiness.py
python scripts\audit_sgaf_claim_language.py
```

Expected readiness report:

- `reports/tables/table_sgaf_package_readiness.md`
- `reports/tables/table_sgaf_external_candidate_acquisition_audit.md`
- `reports/tables/table_sgaf_external_dataset_choice_audit.md`
- `reports/tables/table_sgaf_external_validation_manifest_audit.md`
- `reports/tables/table_sgaf_external_data_readiness_audit.md`
- `reports/tables/table_sgaf_external_run_readiness_audit.md`
- `reports/tables/table_sgaf_external_validation_result_gates.md`
- `reports/tables/table_sgaf_frozen_protocol_readiness.md`
- `reports/tables/table_sgaf_claim_language_audit.md`
- `runs/fusion/frozen_external_validation_readiness/package_readiness.csv`
- `runs/fusion/frozen_external_validation_readiness/package_readiness.json`
- `runs/fusion/frozen_external_validation_readiness/external_candidate_acquisition_audit.csv`
- `runs/fusion/frozen_external_validation_readiness/external_candidate_acquisition_audit.json`
- `runs/fusion/frozen_external_validation_readiness/external_dataset_choice_audit.csv`
- `runs/fusion/frozen_external_validation_readiness/external_dataset_choice_audit.json`
- `runs/fusion/frozen_external_validation_readiness/external_validation_manifest_audit.csv`
- `runs/fusion/frozen_external_validation_readiness/external_validation_manifest_audit.json`
- `runs/fusion/frozen_external_validation_readiness/external_data_readiness_audit.csv`
- `runs/fusion/frozen_external_validation_readiness/external_data_readiness_audit.json`
- `runs/fusion/frozen_external_validation_readiness/external_run_readiness_audit.csv`
- `runs/fusion/frozen_external_validation_readiness/external_run_readiness_audit.json`
- `runs/fusion/frozen_external_validation_readiness/external_validation_result_gates.csv`
- `runs/fusion/frozen_external_validation_readiness/external_validation_result_gates.json`
- `runs/fusion/frozen_external_validation_readiness/frozen_protocol_readiness.csv`
- `runs/fusion/frozen_external_validation_readiness/frozen_protocol_readiness.json`
- `runs/fusion/frozen_external_validation_readiness/claim_language_audit.csv`
- `runs/fusion/frozen_external_validation_readiness/claim_language_audit.json`

1. Choose the held-out dataset/batch with `reports/sgaf_external_validation_candidate_matrix.md`.
2. Create a candidate checklist with `python scripts\init_sgaf_external_candidate_acquisition.py --preset <preset> ...` or fill `configs/sgaf_external_candidate_acquisition.template.json` manually.
3. For BEIR presets, download/extract with `python scripts\download_sgaf_external_beir_candidate.py --preset <preset>`.
4. Audit the acquired candidate with `python scripts\audit_sgaf_external_candidate_acquisition.py <candidate_acquisition.json>`.
5. If needed, convert a local BEIR-format folder with `python scripts\prepare_sgaf_external_beir_dataset.py ...`.
6. Create a held-out run manifest with `python scripts\init_sgaf_external_validation_manifest.py ...`.
7. Audit dataset choice with `python scripts\audit_sgaf_external_dataset_choice.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json`.
8. Validate the filled manifest with `python scripts\audit_sgaf_external_validation_manifest.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json`.
9. Audit local data readiness with `python scripts\audit_sgaf_external_data_readiness.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json`.
10. Generate baseline rankings with `python scripts\run_sgaf_external_baseline_retrieval.py ...`, then generate SGAF rankings with `python scripts\run_sgaf_external_frozen_runs.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json`. Add `--include-p4` only for the appendix-only P4 residual fallback check.
11. Audit generated ranking CSV shape and query coverage with `python scripts\audit_sgaf_external_run_readiness.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json`.
12. Before scoring, rerun data readiness with `--require-qrels`.
13. Score frozen rankings with `python scripts\score_sgaf_external_validation.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --qrels <qrels_path> --cost-guard <pass_or_waiting>`.
14. Compute transfer average if the validation batch contains multiple non-SciFact datasets.
15. Run paired bootstrap or paired randomization tests over per-query nDCG@10.
16. Report duplicate/canonical-document sensitivity if the corpus can contain repeated evidence units.
17. Compare cost against BGE-base and current adaptive SGAF.
18. Apply `python scripts\audit_sgaf_external_validation_results.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json`.
19. Decide claim level using the gates below.

## Pass/Fail Gates

| Gate | Required evidence | Pass criterion | Failure action |
|---|---|---|---|
| Source preservation | SciFact-like held-out batch or duplicate-filtered SciFact if available | P3 loss vs BGE-small <= `0.005` nDCG@10 | demote P3; report B5 only |
| B5 transfer recovery | external transfer set | B5 transfer delta vs BGE-base >= `-0.002` | weaken B5 claim to diagnostic |
| P3 contribution | external transfer set | P3 transfer delta vs B5 > `0` and no dataset delta below `-0.002` | keep P3 exploratory only |
| P3 vs BGE-base | external transfer set | P3 transfer delta vs BGE-base > `0`; significance preferred but not mandatory for "directional" claim | avoid "beats BGE-base" wording |
| P4 promotion | external transfer set | P4 improves over P3 on transfer average, no dataset delta below `-0.002`, and at least two transfer datasets are significant or one larger new dataset is significant | keep P4 appendix-only |
| Cost guard | all external runs | P3/P4 added online cost remains negligible relative to CrossEncoder/LLM reranking | move method to cost/ablation only |

## Reporting Table

Every external validation report should include this contribution table:

| Step | Method | SciFact-like nDCG@10 | Transfer Avg | Delta vs previous | Delta vs BGE-base transfer | Cost note | Claim status |
|---|---|---:|---:|---:|---:|---|---|
| 0 | BGE-base generalist | TBD | TBD | - | +0.0000 | dense baseline | baseline |
| 1 | Frozen B5 mode switch | TBD | TBD | TBD | TBD | cheap controller | pass/fail |
| 2 | Frozen P3 rank-window smoothing | TBD | TBD | TBD | TBD | cheap post-retrieval | pass/fail |
| 3 | Optional P4 residual fallback | TBD | TBD | TBD | TBD | cheap query fallback | appendix/pass/fail |

## Allowed Claims

| Evidence level | Allowed wording |
|---|---|
| Current project datasets only | "Frozen P3 is the strongest current experimental candidate." |
| External validation passes B5 but not P3 | "The mode-switch mechanism preserves specialist behavior and recovers BGE-base transfer under shift." |
| External validation passes P3 directionally | "P3 provides a low-cost post-retrieval improvement over the frozen mode switch on the evaluated transfer data." |
| External validation passes P3 with significance | "P3 improves transfer retrieval over the frozen mode-switch baseline while preserving source performance." |
| External validation passes P4 gate | "A residual specialist fallback can add further gains after P3." |

Avoid these claims unless a larger external validation suite proves them:

- "SGAF universally beats BGE-base."
- "P3 is conclusively better than BGE-base in general."
- "P4 is the main method."
- "The chosen thresholds are globally optimal."

## Current Frozen Baseline For Comparison

| Method | Avg nDCG@10 | Transfer Avg | Transfer delta vs BGE-base | SciFact nDCG@10 |
|---|---:|---:|---:|---:|
| BGE-base generalist | 0.4282 | 0.3251 | +0.0000 | 0.7376 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | -0.0001 | 0.8218 |
| Frozen P3 rank-window smoothing SGAF | 0.4524 | 0.3293 | +0.0042 | 0.8218 |
| Optional P4 residual specialist fallback | 0.4540 | 0.3314 | +0.0063 | 0.8218 |

Interpretation: B5 is the core mechanism, P3 is the main candidate, and P4 remains appendix-only until it passes the promotion gate above.
