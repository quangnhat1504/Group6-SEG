# SGAF External Run Readiness Audit

This audit checks whether generated external-validation ranking CSVs are shape- and coverage-ready before scoring. It does not inspect qrels, compute metrics, run retrieval, or tune parameters.

Manifest: `configs/sgaf_external_validation_manifest.template.json`
Overall status: `pass`

| Gate | Status | Evidence | Action |
|---|---|---|---|
| Run outputs declared | pass | all expected run output keys present | restore planned_outputs run paths in the validation manifest |
| Template mode | pass | template placeholders allowed | continue |

## Decision

- `pass` means all required ranking CSVs can be safely handed to the scorer.
- Missing P4 can pass only with `--allow-missing-optional-p4`; keep P4 appendix-only/waiting in that case.
- Any missing query, duplicate rank, duplicate query-doc pair, or nonnumeric rank/score should be fixed before scoring.
