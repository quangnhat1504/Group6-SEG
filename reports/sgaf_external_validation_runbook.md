# SGAF External Validation Runbook

This runbook turns the frozen SGAF protocol into an execution checklist for a new held-out batch or dataset. It is intentionally operational: what to prepare, what to run, where to put artifacts, and what decision to make. It must not be used to retune B5, P3, or P4.

Authoritative gates:

- `reports/sgaf_frozen_external_validation_protocol.md`
- `reports/sgaf_external_validation_candidate_matrix.md`
- `reports/tables/table_sgaf_frozen_protocol_readiness.md`
- `reports/tables/table_sgaf_claim_language_audit.md`

## 0. Start Conditions

Proceed only if all are true:

| Check | Requirement |
|---|---|
| Dataset identity | dataset/batch name and split are fixed |
| Labels | labels are hidden until rankings are generated, or labels are not used for parameter selection |
| Corpus | corpus snapshot is immutable for this run |
| Frozen package | readiness audit passes before new rankings |
| Claim language | claim-language audit has zero unsafe overclaim hits |

Pre-run command:

```powershell
pytest -q tests\test_sgaf_external_validation_harness.py
python scripts\audit_sgaf_package_readiness.py
python scripts\audit_sgaf_external_candidate_acquisition.py --allow-template
python scripts\audit_sgaf_external_dataset_choice.py --allow-template
python scripts\audit_sgaf_external_validation_manifest.py --allow-template
python scripts\audit_sgaf_external_data_readiness.py --allow-template
python scripts\audit_sgaf_external_run_readiness.py --allow-template
python scripts\audit_sgaf_frozen_protocol_readiness.py
python scripts\audit_sgaf_claim_language.py
```

`audit_sgaf_package_readiness.py` is the aggregate gate. It reruns the focused audits and also checks the paper PDF/log, label references, manifest template, handoff checklist, and navigation links. `audit_sgaf_external_validation_manifest.py --allow-template` validates that the repository template still matches the frozen contract.

## 1. Create Validation Manifest

Choose the dataset/batch using `reports/sgaf_external_validation_candidate_matrix.md` before creating the manifest. Current NFCorpus/FiQA/SciDocs runs are allowed for smoke tests only, not for stronger external-validation claims.

Before data preparation, create a candidate acquisition checklist from a preset:

```powershell
python scripts\init_sgaf_external_candidate_acquisition.py `
  --preset trec_covid_beir `
  --local-source-dir raw\external\trec_covid_beir
```

Available presets: `trec_covid_beir`, `climate_fever`, and `arguana`. The command writes `runs/fusion/external_validation/<candidate_slug>/candidate_acquisition.json`. You can also copy `configs/sgaf_external_candidate_acquisition.template.json` manually for a custom candidate.

If the candidate is a BEIR preset, download and extract the raw BEIR folder:

```powershell
python scripts\download_sgaf_external_beir_candidate.py `
  --preset trec_covid_beir
```

This writes `raw/external/<candidate_slug>/sgaf_external_download_manifest.json` and refreshes the acquisition checklist with `acquisition_status=downloaded`.

Then audit it:

```powershell
python scripts\audit_sgaf_external_candidate_acquisition.py runs\fusion\external_validation\<dataset_slug>\candidate_acquisition.json
```

This gate must pass before manifest initialization. It verifies source identity, license/terms reference, local source directory, expected corpus/query/qrels files, fixed split, and the label-order guard.

If the candidate is already available as a local BEIR-format folder, prepare repository-native files first:

```powershell
python scripts\prepare_sgaf_external_beir_dataset.py `
  --input-dir <local_beir_dataset_dir> `
  --dataset-slug <dataset_slug> `
  --split test `
  --source-name <source_name>
```

This writes `data/external/<dataset_slug>/<split>_documents.jsonl`, `<split>_queries.jsonl`, `<split>_qrels.csv`, and `sgaf_external_data_prep_manifest.json`.

Create a filled manifest from the frozen template:

```powershell
python scripts\init_sgaf_external_validation_manifest.py `
  --dataset-slug <dataset_slug> `
  --corpus-snapshot <path_hash_or_date> `
  --query-snapshot <path_hash_or_date> `
  --qrels-snapshot hidden_until_after_ranking `
  --split-policy <official_test_or_frozen_heldout>
```

It writes:

```text
runs/fusion/external_validation/<dataset_slug>/validation_manifest.json
```

The manifest is the run contract. If a field changes after labels are inspected, the validation is no longer clean and must be reported as exploratory. If you copy the template manually instead, fill every `TODO` field before running retrieval.

Validate the filled manifest before retrieval:

```powershell
python scripts\audit_sgaf_external_dataset_choice.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
python scripts\audit_sgaf_external_validation_manifest.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
python scripts\audit_sgaf_external_data_readiness.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
```

After all expected rankings are generated, run the shape and coverage gate before scoring:

```powershell
python scripts\audit_sgaf_external_run_readiness.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
```

If P4 is intentionally not evaluated, add `--allow-missing-optional-p4` and keep P4 as appendix-only/waiting.

After all expected rankings and report files are generated, rerun with:

```powershell
python scripts\audit_sgaf_external_validation_manifest.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --require-existing-outputs
python scripts\audit_sgaf_external_validation_results.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
```

If P4 is intentionally not evaluated, add `--allow-missing-optional-p4` to the post-output manifest audit and keep P4 as appendix-only/waiting.

Minimum manifest fields:

| Field | Meaning |
|---|---|
| `dataset_slug` | short lowercase ID, for example `trec_covid_heldout` |
| `corpus_snapshot` | path/hash/date for corpus version |
| `query_snapshot` | path/hash/date for query version |
| `qrels_snapshot` | path/hash/date for labels, or `hidden_until_after_ranking` |
| `split_policy` | official test, held-out batch, or frozen manual split |
| `frozen_recipes` | B5/P3/P4 parameters copied from protocol |
| `planned_outputs` | exact output paths for baseline and SGAF runs |

## 2. Generate Baseline Runs

Only start ranking after data readiness passes. If qrels are hidden, this audit may pass before ranking with `qrels_snapshot=hidden_until_after_ranking`; rerun it with `--require-qrels` before scoring once labels are available.

Generate these rankings before looking at metrics:

| Run | Required output |
|---|---|
| BM25 | `runs/fusion/external_validation/<dataset_slug>/bm25.csv` |
| BGE-small specialist | `runs/fusion/external_validation/<dataset_slug>/bge_small_specialist.csv` |
| BGE-base generalist | `runs/fusion/external_validation/<dataset_slug>/bge_base_generalist.csv` |

Use the manifest-based baseline runner:

```powershell
python scripts\run_sgaf_external_baseline_retrieval.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --output-key bm25
python scripts\run_sgaf_external_baseline_retrieval.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --output-key bge_small_specialist --model-path runs\finetuned\bge-small-final
python scripts\run_sgaf_external_baseline_retrieval.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --output-key bge_base_generalist --model-path BAAI/bge-base-en-v1.5
```

This runner writes ranking files and sidecar `.manifest.json` files. It does not load qrels or compute metrics.

Each run must contain at least:

```text
query_id, doc_id, rank, score
```

If existing scripts emit additional columns, keep them. Do not normalize or calibrate scores using validation labels.

## 3. Generate Frozen SGAF Runs

Generate these runs with unchanged recipes:

| Run | Frozen recipe | Required output |
|---|---|---|
| Current adaptive SGAF | current predecessor only | `current_adaptive_sgaf.csv` |
| Frozen B5 | threshold `2.0`, gain `6.0`, cap `1.0`, `C=0.1` | `frozen_b5_sgaf.csv` |
| Frozen P3 | B5 + `window=20`, `alpha=0.10`, `rrf_k=60` | `frozen_p3_sgaf.csv` |
| Optional P4 | P3 + `small_minus_base_top`, fraction `0.10` | `optional_p4_sgaf.csv` |

Use the manifest-based frozen SGAF runner for the required non-P4 runs:

```powershell
python scripts\run_sgaf_external_frozen_runs.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
```

This runner consumes the three baseline ranking files, retrains the frozen SciFact trainfit rescue classifier from the local frozen state, writes `current_adaptive_sgaf.csv`, `frozen_b5_sgaf.csv`, and `frozen_p3_sgaf.csv`, and writes sidecar manifests. It does not load qrels, compute metrics, or select external-dataset parameters.

To evaluate the appendix-only residual fallback, add:

```powershell
python scripts\run_sgaf_external_frozen_runs.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --include-p4
```

`--include-p4` writes `optional_p4_sgaf.csv` using the frozen feature `small_minus_base_top` and fraction `0.10`. Keep it out of the main claim unless the P4 promotion gate passes.

Required per-query diagnostic columns for SGAF runs:

```text
query_id, mode, shift_score, coverage, selected_by_p4
```

Use `selected_by_p4=false` for non-P4 runs. If a script cannot emit these columns yet, add the columns before treating the run as externally valid.

## 4. Score After Rankings Are Frozen

After all rankings exist, attach qrels and compute:

| Metric | Scope |
|---|---|
| nDCG@10 | primary metric, per query and aggregate |
| Recall@10 / Recall@100 | secondary retrieval check |
| MRR@10 | secondary ranking check |
| transfer average | if multiple non-SciFact datasets are included |
| duplicate/canonical sensitivity | if corpus has repeated evidence units |
| latency/cost | online cost relative to BGE-base and CE/LLM rerankers |

Use the scorer after qrels are available and all ranking CSVs are frozen:

```powershell
python scripts\audit_sgaf_external_data_readiness.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json --require-qrels
python scripts\audit_sgaf_external_run_readiness.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json
python scripts\score_sgaf_external_validation.py runs\fusion\external_validation\<dataset_slug>\validation_manifest.json `
  --qrels <qrels_path> `
  --cost-guard <pass_or_waiting>
```

Use `--source-like` if the external batch is SciFact-like and should count for the source-preservation gate. Use `--allow-missing-optional-p4` only when P4 is intentionally not evaluated and should remain appendix-only/waiting.

Recommended output paths:

```text
runs/fusion/external_validation/<dataset_slug>/external_validation_rows.csv
runs/fusion/external_validation/<dataset_slug>/external_validation_summary.csv
runs/fusion/external_validation/<dataset_slug>/external_validation_significance.csv
reports/tables/table_sgaf_external_validation_<dataset_slug>.md
```

Minimum machine-readable fields for result-gate audit:

| File | Required fields |
|---|---|
| `external_validation_summary.csv` | `method`, `transfer_avg_ndcg@10`, `cost_guard`; include `scifact_ndcg@10` or `source_ndcg@10` when a source-like batch is available |
| `external_validation_rows.csv` | per-dataset or held-out rows with `p3_delta_vs_b5` and, if P4 is evaluated, `p4_delta_vs_p3` |
| `external_validation_significance.csv` | `baseline`, `significant`; include rows for P4 vs P3 if P4 promotion is evaluated |

## 5. Apply Decision Gates

Use the protocol gates exactly:

| Gate | Pass condition | If fail |
|---|---|---|
| Source preservation | P3 source-like loss vs BGE-small <= `0.005` | demote P3, report B5 only |
| B5 transfer recovery | B5 transfer delta vs BGE-base >= `-0.002` | weaken B5 to diagnostic |
| P3 contribution | P3 transfer delta vs B5 > `0`, no dataset below `-0.002` | keep P3 exploratory |
| P3 vs BGE-base | P3 transfer delta vs BGE-base > `0` | avoid BGE-base-beating wording |
| P4 promotion | P4 > P3, no dataset below `-0.002`, stronger significance than current evidence | keep P4 appendix-only |
| Cost guard | P3/P4 online cost remains negligible | move method to cost ablation |

The result-gate audit writes:

```text
runs/fusion/frozen_external_validation_readiness/external_validation_result_gates.csv
runs/fusion/frozen_external_validation_readiness/external_validation_result_gates.json
reports/tables/table_sgaf_external_validation_result_gates.md
```

## 6. Required Report Shape

Every external validation report must include:

1. Dataset snapshot and split policy.
2. Frozen recipe block copied from the manifest.
3. Baseline table: BM25, BGE-small, BGE-base.
4. Contribution table: BGE-base -> B5 -> P3 -> P4.
5. Per-dataset gate table with pass/fail status.
6. Significance table over per-query nDCG@10.
7. Claim-language decision: what wording is now allowed and what remains forbidden.

Do not bury failed gates. A failed gate is a result, not a cleanup item.

## 7. Post-Run Claim Update

Run:

```powershell
python scripts\audit_sgaf_claim_language.py
```

Then update:

- `reports/tables/table_final_sgaf_synthesis.md`
- `reports/tables/table_sgaf_claim_audit.md`
- `wiki/plan.md`
- `wiki/overview.md`
- paper sections only after the report artifacts are stable

Allowed outcomes:

| Outcome | Paper direction |
|---|---|
| B5 passes, P3 fails | main method = B5, P3 appendix |
| B5 and P3 pass directionally | main method = P3 with external directional caveat |
| B5 and P3 pass significantly | stronger P3 claim allowed |
| P4 fails promotion | P4 remains appendix-only |
| P4 passes promotion | P4 can be discussed as a residual extension, not automatically renamed main method |

## 8. Stop Rules

Stop and label the run exploratory if:

- any SGAF parameter changes after labels are inspected;
- any dataset-specific threshold, alpha, fraction, or feature is selected;
- qrels are used to choose the model or tune retrieval depth;
- output paths are overwritten without preserving the old manifest;
- manifest audit fails before retrieval or after expected outputs are generated;
- result-gate audit fails after summary/significance files are generated;
- claim language audit fails with unsafe overclaim hits.

The next clean run must start with a new manifest and frozen settings.
