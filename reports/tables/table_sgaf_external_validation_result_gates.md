# SGAF External Validation Result Gates

This report applies the frozen SGAF pass/fail gates to an external-validation summary. It does not run retrieval, score qrels, or tune parameters.

Manifest: `configs/sgaf_external_validation_manifest.template.json`
Overall status: `waiting`

| Gate | Status | Evidence | Action |
|---|---|---|---|
| External result files | waiting | missing: summary, significance, rows | generate external rankings, rows, summary, and significance before applying result gates |

## Decision

- `pass` means all required evaluated gates passed.
- `waiting` means the external run has not produced enough evidence for that gate yet.
- `fail` means claim language must be weakened according to the action column.
