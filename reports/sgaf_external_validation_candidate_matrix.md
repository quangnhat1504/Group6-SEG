# SGAF External Validation Candidate Matrix

Purpose: choose the next held-out dataset or batch for frozen SGAF validation without retuning B5, P3, or P4. This is a planning gate, not a result claim.

## Current Local Evidence

The repository currently has these local datasets:

| Dataset | Test qrels queries | Query file rows | Corpus rows | Status for claim promotion |
|---|---:|---:|---:|---|
| SciFact | 300 | 300 | 5183 | source/development dataset; useful for source preservation only |
| NFCorpus | 323 | 323 | 3633 | already used in B5/P3/P4 transfer exploration; not clean external validation |
| FiQA | 648 | 6648 | 57638 | already used in B5/P3/P4 transfer exploration; not clean external validation |
| SciDocs | 1000 | 1000 | 25657 | already used in B5/P3/P4 transfer exploration; not clean external validation |

Decision: the current local datasets can still be used for smoke tests, regression tests, and report reproduction. They should not be used to strengthen the claim beyond "current evaluated datasets" because B5/P3/P4 were developed after seeing this set.

## Selection Criteria

| Criterion | Requirement | Why it matters |
|---|---|---|
| External cleanliness | Dataset/batch was not used to choose B5 threshold/gain, P3 window/alpha, or P4 feature/fraction | prevents post-hoc tuning claims |
| Fixed split | official test split or frozen held-out batch before scoring | prevents moving target evaluation |
| Qrels availability | relevance labels available only after rankings are frozen, or at least not used for recipe decisions | protects validation order |
| Baseline reproducibility | BM25, BGE-small specialist, and BGE-base can be generated under the manifest contract | required by result gates |
| Domain value | either source-like scientific retrieval or a transfer domain with different vocabulary/query style | tests the specialist/generalist story |
| Size | enough queries for stable paired bootstrap; prefer at least a few hundred queries | reduces noise in P3/P4 gates |
| Cost | can run dense/BM25 rankings without CrossEncoder/LLM reranking | keeps the method aligned with cheap post-retrieval direction |

## Candidate Matrix

| Candidate | Role | Expected validation value | Setup cost | Main risk | Decision |
|---|---|---|---|---|---|
| New SciFact-like batch | source preservation | strongest test of "do not lose specialist behavior" | medium/high, requires new labels or curated held-out claims | hard to obtain clean labels quickly | best if source-preservation claim is the priority |
| TREC-COVID style biomedical retrieval | transfer under biomedical/scientific vocabulary | tests whether B5 generalist fallback holds on scientific but non-SciFact evidence | medium, needs data prep plus BM25/dense runs | may overlap semantically with NFCorpus-style behavior | high-priority external transfer candidate |
| Climate-FEVER style claim retrieval | claim/evidence transfer | closest task shape to SciFact while changing topic domain | medium, needs claim/corpus/qrels conversion | evidence structure may differ from abstract-level SciFact | high-priority claim-transfer candidate |
| ArguAna style argument retrieval | hard semantic transfer | stresses dense/generalist behavior with different relevance semantics | medium | may be too far from scientific retrieval claim | useful robustness candidate, not first promotion run |
| CQADupStack-style forum retrieval | broad lexical/semantic transfer | tests generality outside scientific claims | medium/high due to multiple subtasks | may make source-specialist story less central | second-wave robustness |
| DBPedia/entity retrieval | entity-heavy transfer | probes lexical/entity matching and BM25 interaction | medium | not claim/evidence retrieval | diagnostic, not main promotion candidate |
| Reuse NFCorpus/FiQA/SciDocs | smoke/regression only | verifies harness and reproduces current gates | low | contaminated for external claims | allowed only as smoke or appendix reproduction |

## Recommended Sequence

1. Run one **source-like** validation if a clean SciFact-like batch can be created without tuning.
2. Run one **claim-transfer** validation, preferably Climate-FEVER style if conversion is practical.
3. Run one **biomedical transfer** validation, preferably TREC-COVID style if setup is practical.
4. Keep ArguAna/CQADupStack/DBPedia for second-wave robustness after the first frozen external result.

Minimum first external package:

| Slot | Candidate type | Required output |
|---|---|---|
| Source slot | new SciFact-like batch or duplicate-free source-like batch | source preservation gate |
| Transfer slot | one new non-project dataset | B5 recovery, P3 contribution, P3 vs BGE-base |
| Optional residual slot | same transfer dataset with optional P4 run | P4 promotion gate remains appendix-only unless passed |

## Stop Conditions

Stop and report the run as exploratory if:

- any candidate is chosen because its labels are known to favor P3/P4;
- B5/P3/P4 parameters are changed after selecting the dataset;
- current NFCorpus/FiQA/SciDocs are presented as new external validation;
- qrels are used to decide retrieval depth, alpha, threshold, fallback fraction, or candidate dataset after ranking metrics are inspected.

## Operational Next Step

For the chosen candidate:

```powershell
python scripts\init_sgaf_external_candidate_acquisition.py `
  --preset <trec_covid_beir_or_climate_fever_or_arguana> `
  --local-source-dir raw\external\<dataset_slug>

python scripts\download_sgaf_external_beir_candidate.py `
  --preset <trec_covid_beir_or_climate_fever_or_arguana>

python scripts\audit_sgaf_external_candidate_acquisition.py `
  runs\fusion\external_validation\<dataset_slug>\candidate_acquisition.json

python scripts\init_sgaf_external_validation_manifest.py `
  --dataset-slug <dataset_slug> `
  --corpus-snapshot <path_hash_or_date> `
  --query-snapshot <path_hash_or_date> `
  --qrels-snapshot hidden_until_after_ranking `
  --split-policy <official_test_or_frozen_heldout>
```

Then follow `reports/sgaf_external_validation_runbook.md`.
