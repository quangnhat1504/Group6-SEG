# SGAF External Candidate Acquisition Audit

This audit checks whether a proposed external candidate is locally acquired and clean enough to move into SGAF manifest initialization. It does not download data, run retrieval, inspect qrels metrics, or tune parameters.

Checklist: `configs/sgaf_external_candidate_acquisition.template.json`
Overall status: `pass`

| Gate | Status | Evidence | Action |
|---|---|---|---|
| Checklist fields present | pass | all required fields present | restore the acquisition checklist schema |
| Template mode | pass | template placeholders allowed | continue |

## Decision

- `pass` means the candidate can move to BEIR/local data preparation and validation manifest initialization.
- Project datasets can pass only for `smoke_only` scope.
- Public qrels are acceptable only if they are not used for candidate or parameter selection before rankings are frozen.
