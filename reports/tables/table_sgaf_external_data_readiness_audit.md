# SGAF External Data Readiness Audit

This audit checks local corpus/query/qrels file readiness for a filled SGAF external-validation manifest. It does not download data, run retrieval, score rankings, or tune parameters.

Manifest: `configs/sgaf_external_validation_manifest.template.json`
Overall status: `pass`

| Gate | Status | Evidence | Action |
|---|---|---|---|
| Data contract present | pass | all data contract keys present | restore data_contract in validation manifest |
| Template mode | pass | template placeholders allowed | continue |

## Decision

- `pass` means the local data files are ready for the requested phase.
- Before ranking, `qrels_snapshot=hidden_until_after_ranking` is allowed.
- Before scoring, rerun with `--require-qrels` so qrels parse successfully.
