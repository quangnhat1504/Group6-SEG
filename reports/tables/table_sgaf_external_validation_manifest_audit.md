# SGAF External Validation Manifest Audit

This audit validates the manifest contract for a future frozen external-validation run. It does not run retrieval, score rankings, or tune parameters.

Manifest: `configs/sgaf_external_validation_manifest.template.json`
Overall status: `pass`

| Gate | Status | Evidence | Action |
|---|---|---|---|
| Manifest file | pass | configs/sgaf_external_validation_manifest.template.json | continue |
| JSON parse | pass | json parsed | continue |
| Top-level schema | pass | all required top-level keys present | continue |
| Manifest identity | pass | dataset_slug=TODO_lowercase_dataset_or_batch_id, created_at=TODO_YYYY-MM-DD | continue |
| Candidate contract | pass | candidate contract template present | continue |
| Data contract | pass | data contract present and labels are not used before ranking | continue |
| Frozen recipe lock | pass | frozen B5/P3/P4 values match protocol | continue |
| Decision gate lock | pass | decision gates match protocol | continue |
| Planned output contract | pass | planned outputs are under runs/fusion/external_validation/TODO_dataset_slug | continue |
| Pre-run audit list | pass | all required pre-run audits listed | continue |

## Decision

- If this audit passes before ranking, the run has a valid frozen manifest contract.
- If this audit passes with `--require-existing-outputs`, rankings and report files exist at the manifest paths.
- Passing this audit does not validate retrieval quality; it only validates run hygiene.
