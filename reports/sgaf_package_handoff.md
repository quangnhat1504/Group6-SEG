# SGAF Package Handoff Checklist

This checklist summarizes the current SGAF package state for paper-language review, future external validation, or a clean commit. It is not a method proposal and it must not be used to change B5/P3/P4 parameters.

## Current Decision

| Item | Status |
|---|---|
| Main current candidate | Frozen P3 Rank-Window Smoothing SGAF |
| Core mechanism | Frozen B5 Mode-Switch SGAF |
| Optional extension | P4 Residual Specialist Fallback, appendix-only |
| Stronger claim blocker | no new external held-out dataset/batch yet |
| Package gate | `python scripts\audit_sgaf_package_readiness.py` |

Current numeric anchor:

| Method | Transfer Avg | Delta vs BGE-base transfer | SciFact nDCG@10 |
|---|---:|---:|---:|
| BGE-base generalist | 0.3251 | +0.0000 | 0.7376 |
| Frozen B5 mode-switch SGAF | 0.3249 | -0.0001 | 0.8218 |
| Frozen P3 rank-window smoothing SGAF | 0.3293 | +0.0042 | 0.8218 |
| Optional P4 residual fallback | 0.3314 | +0.0063 | 0.8218 |

Interpretation: P3 is the main current candidate because it improves over B5 and clears current BGE-base transfer average while preserving SciFact. P4 is positive but stays appendix-only because current evidence has only one significant transfer dataset versus P3.

## Commands Before Handoff

Run:

```powershell
python scripts\audit_sgaf_package_readiness.py
```

Optional focused checks:

```powershell
python scripts\audit_sgaf_frozen_protocol_readiness.py
python scripts\audit_sgaf_claim_language.py
```

Expected pass artifacts:

- `reports/tables/table_sgaf_package_readiness.md`
- `reports/tables/table_sgaf_frozen_protocol_readiness.md`
- `reports/tables/table_sgaf_claim_language_audit.md`

## Source Files To Keep In Review Scope

Core package docs:

- `reports/sgaf_frozen_external_validation_protocol.md`
- `reports/sgaf_external_validation_candidate_matrix.md`
- `reports/sgaf_external_validation_runbook.md`
- `reports/sgaf_package_handoff.md`
- `configs/sgaf_external_validation_manifest.template.json`

Audit scripts:

- `scripts/audit_sgaf_package_readiness.py`
- `scripts/audit_sgaf_frozen_protocol_readiness.py`
- `scripts/audit_sgaf_claim_language.py`

Audit reports:

- `reports/tables/table_sgaf_package_readiness.md`
- `reports/tables/table_sgaf_frozen_protocol_readiness.md`
- `reports/tables/table_sgaf_claim_language_audit.md`

Navigation:

- `wiki/index.md`
- `wiki/plan.md`

Paper-facing files changed during SGAF packaging:

- `paper/sections/results.tex`
- `paper/tables/table10_p4_residual_fallback.tex`
- `paper/sections/appendix.tex`
- `paper/sections/limitations.tex`
- `paper/sections/conclusion.tex`

## Generated Or Ignored Artifacts

Keep these available locally for audit/build, but do not track unless submission packaging explicitly requires them:

- `paper/main.pdf`
- `paper/main.log`
- `paper/main.aux`
- `paper/main.bbl`
- `paper/main.blg`
- `paper/main.toc`
- `runs/fusion/frozen_external_validation_readiness/*.json`
- `runs/fusion/frozen_external_validation_readiness/*.csv`
- `_tmp_pdf_visual_*/`

Current `.gitignore` intentionally hides LaTeX build outputs, `/runs/`, PDFs, and temporary PDF visual inspection folders.

## Claim Boundaries

Allowed:

- Frozen P3 is the strongest current experimental candidate.
- Frozen B5 is the cleaner mode-switch core.
- P4 is a positive optional appendix diagnostic.
- B5/P3 preserve the SciFact specialist while improving transfer behavior on the current evaluated datasets.

Avoid:

- SGAF universally beats BGE-base.
- P3 is conclusively better than BGE-base in general.
- P4 is the main method.
- B5/P3/P4 thresholds are globally optimal.

## Next Action If New Data Appears

Use:

- `reports/sgaf_external_validation_candidate_matrix.md`
- `reports/sgaf_external_validation_runbook.md`
- `configs/sgaf_external_validation_manifest.template.json`

Do not change:

- B5 threshold/gain/cap/C.
- P3 window/alpha/rrf_k.
- P4 feature/fraction.

Any parameter change after seeing labels makes the run exploratory.

## Current Package Status

As of the latest package audit:

- package readiness: pass;
- frozen protocol readiness: pass;
- claim-language safety: pass;
- PDF artifact exists and LaTeX log has no source-blocking error;
- remaining warning is local MiKTeX/Babel Vietnamese hyphenation only.
