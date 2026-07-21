# SGAF Claim Language Audit

This audit checks whether current paper/wiki/report language stays within the frozen SGAF evidence gates. It distinguishes unsafe overclaims from deliberately quoted rejected/avoid claims.

## Summary

| Check | Count | Status |
|---|---:|---|
| Required cautious framings | 4/4 | pass |
| Unsafe overclaim hits | 0 | pass |
| Allowed rejected/avoid-claim examples | 16 | info |

## Required Framing

| Requirement | Status | Evidence | Action |
|---|---|---|---|
| P3 framed as experimental candidate | pass | Frozen P3 Rank-Window Smoothing SGAF** is the strongest current experimental candidate | continue |
| P3 external validation caveat | pass | P3, và optional P4 nếu dùng, trên một held-out batch | continue |
| P4 appendix or optional framing | pass | P4 adds `+0.0021` transfer over P3 in LOTO, but evidence is weaker than P3 and should not replace the main | continue |
| BGE-base universal-outperformance caveat | pass | universal retriever | Rejected | Transfer average is 0.3098 vs BGE-base | continue |

## Unsafe Hits

| Kind | Status | Location | Evidence | Action |
|---|---|---|---|---|
| none | pass | - | no unsafe overclaim wording found | continue |

## Allowed Rejected/Avoid Examples

| Kind | Location | Evidence |
|---|---|---|
| p4_main_method | `reports/tables/table_final_sgaf_synthesis.md:44` | | Adaptive coverage | Useful predecessor | Uncertainty-shift improves over fixed A3, but still trails BGE-base transfer by -0.0152 average. | | BGE-base mode switch | Core candi... |
| p4_main_method | `reports/tables/table_final_sgaf_synthesis.md:65` | | 2 | P3 rank-window smoothing inside generalist-fallback | 0.4524 | 0.3293 | +0.0043 | +0.0042 | 0.8218 | main candidate | | 3 | P4 residual specialist fallback, top 10% confid... |
| globally_optimal_threshold | `reports/tables/table_sgaf_claim_audit.md:60` | Avoid: > The threshold is fully validated as generally optimal. |
| conclusive_bge_base_claim | `reports/tables/table_sgaf_claim_audit.md:64` | Avoid: > Frozen P3 is conclusively better than BGE-base in general. |
| universal_bge_base_outperformance | `reports/sgaf_frozen_external_validation_protocol.md:153` | Avoid these claims unless a larger external validation suite proves them: - "SGAF universally beats BGE-base." - "P3 is conclusively better than BGE-base in general." |
| conclusive_bge_base_claim | `reports/sgaf_frozen_external_validation_protocol.md:154` | Avoid these claims unless a larger external validation suite proves them: - "SGAF universally beats BGE-base." - "P3 is conclusively better than BGE-base in general." - "P4 is t... |
| p4_main_method | `reports/sgaf_frozen_external_validation_protocol.md:155` | - "SGAF universally beats BGE-base." - "P3 is conclusively better than BGE-base in general." - "P4 is the main method." - "The chosen thresholds are globally optimal." |
| globally_optimal_threshold | `reports/sgaf_frozen_external_validation_protocol.md:156` | - "SGAF universally beats BGE-base." - "P3 is conclusively better than BGE-base in general." - "P4 is the main method." - "The chosen thresholds are globally optimal." |
| p4_main_method | `wiki/overview.md:24` | 4. **BGE-small final (SciFact specialist, 33M)** → **0.8188 nDCG@10** on SciFact, but weaker cross-dataset transfer 5. **Frozen B5 Mode-Switch SGAF** → **0.8218** on SciFact and... |
| paper_grade_without_validation | `wiki/plan.md:313` | Completion caveat: - The adaptive coverage formula was selected after Phase 5 exploratory ablation. It is now a predecessor ablation rather than the final candidate; B5/P3 super... |
| universal_bge_base_outperformance | `wiki/plan.md:434` | - Added `reports/tables/table_sgaf_claim_audit.md`. - At that stage, `reports/tables/table_final_sgaf_synthesis.md` promoted Frozen B5 Mode-Switch SGAF over uncertainty-shift ad... |
| paper_grade_without_validation | `wiki/plan.md:529` | | FiQA | NFCorpus, SciDocs | `window=20, alpha=0.10` | +0.0051 | no, `p=0.0938` | | SciDocs | NFCorpus, FiQA | `window=20, alpha=0.10` | +0.0027 | no, `p=0.0670` | - Decision: P... |
| p4_main_method | `wiki/plan.md:551` | | FiQA | 0.3960 | 0.3977 | +0.0016 | +0.0067 | no, `p=0.5408` | | SciDocs | 0.2173 | 0.2198 | +0.0024 | +0.0051 | yes, `p=0.0422` | - Decision: P4 is positive but small. Keep it... |
| conclusive_bge_base_claim | `wiki/plan.md:557` | - Frozen P3 is the strongest current experimental candidate. - Frozen B5 remains the core mode-switch mechanism and the cleaner paper-safe ablation. - Do not present P3 as a gen... |
| p4_main_method | `wiki/plan.md:565` | - Use `reports/sgaf_external_validation_candidate_matrix.md` first, then `reports/sgaf_external_validation_runbook.md`, `configs/sgaf_external_candidate_acquisition.template.jso... |
| universal_bge_base_outperformance | `wiki/concepts/Fine-Tune-BGE-RRF.md:20` | - **BGE-base (109M):** Failed 3× with catastrophic forgetting (−0.0135 to −0.0399). MNRL too aggressive for large pretrained model. Need lower LR + more freeze + different loss ... |

## Decision

- Current claim language is acceptable if all required framings pass and unsafe hits remain zero.
- P3 can be described as the strongest current experimental candidate, but not as a generally validated BGE-base replacement.
- P4 should remain optional/appendix-only until the frozen external promotion gate passes.
