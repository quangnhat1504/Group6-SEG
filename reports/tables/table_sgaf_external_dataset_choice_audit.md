# SGAF External Dataset Choice Audit

This audit checks whether a proposed external-validation dataset choice is clean enough for the intended claim scope. It does not inspect qrels, run retrieval, or tune SGAF parameters.

Manifest: `configs/sgaf_external_validation_manifest.template.json`
Overall status: `pass`

| Gate | Status | Evidence | Action |
|---|---|---|---|
| Candidate contract present | pass | all candidate contract keys present | continue |
| Template mode | pass | template placeholders allowed | continue |

## Decision

- `pass` means the dataset choice is compatible with its declared claim scope.
- `fail` means use smoke/appendix wording or choose a cleaner candidate before running rankings.
