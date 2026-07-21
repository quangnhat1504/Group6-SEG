# Wiki Index

## Overview
- [Overview](overview.md) — living synthesis of the SEG project
- [Plan](plan.md) — next steps roadmap (Phases A-D)

## Core Concepts (Updated July 2026)
- [Specialist-First Cheap Retrieval Repair](concepts/Specialist-First-Cheap-Retrieval-Repair.md) — cheap post-retrieval repair path before expensive specialist-generalist fusion
- [Specialist-Generalist Adaptive Fusion](concepts/Specialist-Generalist-Adaptive-Fusion.md) — new performance direction: query-adaptive fusion of specialist and generalist retrievers
- [BGE](concepts/BGE.md) — BGE-base-en-v1.5, the modern embedding model replacing SciNCL
- [Adaptive RRF](concepts/Adaptive-RRF.md) — Per-query IDF-weighted reciprocal rank fusion
- [nDCG@10](concepts/nDCG@10.md) — Primary evaluation metric (Al-Joofi 2025)
- [Paired T-Test](concepts/Paired-Bootstrap.md) — Significance testing methodology (Shapiro-Wilk + t-test / Wilcoxon)
- [SciFact](concepts/SciFact.md) — Primary scientific claims dataset
- [SciDocs](concepts/SciDocs.md) — Citation prediction dataset (BEIR, 25K docs)
- [FiQA](concepts/FiQA.md) — Financial QA dataset (BEIR, 57K docs)
- [BM25 Retrieval](concepts/BM25-Retrieval.md) — Lexical sparse retrieval baseline
- [Dense Retrieval](concepts/Dense-Retrieval.md) — Semantic embedding retrieval
- [Reciprocal Rank Fusion](concepts/Reciprocal-Rank-Fusion.md) — Hybrid rank combination
- [Fine-Tune BGE RRF](concepts/Fine-Tune-BGE-RRF.md) — Self-supervised embedding fine-tuning via disagreement signal — **successful: 0.7909 NDCG** + Phase 0-7 implementation complete
- [vstash](concepts/vstash.md) — External reference model (Stffens/bge-small-rrf-v3, 0.7707 NDCG)
- [Catastrophic Forgetting](concepts/Catastrophic-Forgetting.md) — Why fine-tuning BGE-base fails
- [MAP@10](concepts/MAP@10.md) — Secondary evaluation metric
- [Downstream-Utility Reranking](concepts/Downstream-Utility-Reranking.md) — Failed LLM-based reranking attempt
- [Semantic Entropy](concepts/Semantic-Entropy.md) — Uncertainty metric

## Historical Concepts (SciNCL-era, mostly superseded by BGE)
- [Selective Reranking](concepts/Selective-Reranking.md) — Uncertainty-gated cross-encoder (was useful for SciNCL, harmful for BGE)
- [Conformal Risk Control](concepts/Conformal-Risk-Control.md) — Principled threshold selection (no longer needed)
- [Query Performance Prediction](concepts/Query-Performance-Prediction.md) — Unsupervised trigger signals
- [CrossEncoder](concepts/CrossEncoder.md) — Pair-wise relevance reranker (degrades BGE pipeline)
- [Query Routing](concepts/Query-Routing.md) — Per-query retrieval method selection
- [Always-Rerank](concepts/Always-Rerank.md) — Static baseline reranking all queries

## Sources
- [Paper — LaTeX Build Instructions](sources/paper-readme.md)
- [Phase 3 Progress — 2026-06-20](sources/phase3-progress-2026-06-20.md)
- [Phase 3 Progress — 2026-06-24](sources/phase3-progress-2026-06-24.md)
- [Push Performance Phase 0-7](sources/push-performance-phase0-3.md) — Full implementation: leak audit + clean protocol + triple quality + curriculum + RRF tuning + cross-dataset + finalize (2026-07-11)
- [June 29 Experiment Session](sources/june-29-experiment-session.md)
- [Validation Experiments](sources/validation-experiments.md)
- [Phase 1-2 Report](sources/phase1-phase2-report.md)
- [Task List](sources/tasks.md)
- [Research Paper — Current Progress](sources/seg-research-paper.md)
- [Research Directions](sources/seg-research-directions.md)
- [Research Deep Plan](sources/seg-research-deep-plan.md)
- [Cheap Post-Retrieval Repair Ablation](sources/cheap-post-retrieval-repair-ablation.md)
- [Cheap Repair Multi-Seed - Dev](../reports/tables/table_cheap_repair_multiseed_dev.md)
- [Cheap Repair Multi-Seed - Test](../reports/tables/table_cheap_repair_multiseed_test.md)
- [Cheap Repair Clean Validation](../reports/tables/table_cheap_repair_clean_validation.md)
- [Cheap Repair Failure Aggregate - Dev](../reports/tables/table_cheap_repair_failure_aggregate_dev.md)
- [Cheap Repair Failure Aggregate - Test](../reports/tables/table_cheap_repair_failure_aggregate_test.md)
- [Cheap Repair Case Profiles - Dev](../reports/tables/table_cheap_repair_case_profiles_dev.md)
- [Cheap Repair Case Profiles - Test](../reports/tables/table_cheap_repair_case_profiles_test.md)
- [Static Ensemble Dev Validation](../reports/tables/table_static_ensemble_dev.md)
- [Query-Adaptive Fusion](../reports/tables/table_query_adaptive_fusion.md)
- [Frozen SGAF Transfer](../reports/tables/table_frozen_sgaf_transfer.md)
- [Frozen SGAF Robustness](../reports/tables/table_frozen_sgaf_robustness.md)
- [Adaptive Coverage SGAF](../reports/tables/table_adaptive_coverage_sgaf.md)
- [Adaptive Coverage Failure Analysis](../reports/tables/table_adaptive_coverage_failure_analysis.md)
- [Final SGAF Synthesis](../reports/tables/table_final_sgaf_synthesis.md)
- [SGAF BGE-base Mode-Switch Plan](../reports/sgaf_bge_base_mode_switch_plan.md)
- [SGAF Mode-Switch Ablation](../reports/tables/table_sgaf_mode_switch_ablation.md)
- [Final B5 SGAF Mode-Switch](../reports/tables/table_final_sgaf_mode_switch.md)
- [SGAF Claim Audit](../reports/tables/table_sgaf_claim_audit.md)
- [SGAF Phase 8 Validation and Ablation Plan](../reports/sgaf_phase8_validation_ablation_plan.md)
- [SGAF Phase 8 Validation and Contribution Tables](../reports/tables/table_sgaf_phase8_validation_ablation.md)
- [SGAF Phase 8C Post-Retrieval Collapse Ablation](../reports/tables/table_sgaf_phase8_post_retrieval_collapse.md)
- [SGAF Phase 8C Rank-Window Smoothing Ablation](../reports/tables/table_sgaf_phase8_rank_window_smoothing.md)
- [SGAF P3 Leave-One-Transfer-Dataset-Out Validation](../reports/tables/table_sgaf_p3_loto_validation.md)
- [Frozen P3 Rank-Window Smoothing SGAF](../reports/tables/table_final_sgaf_p3_smoothing.md)
- [SGAF Frozen External Validation Protocol](../reports/sgaf_frozen_external_validation_protocol.md)
- [SGAF External Validation Candidate Matrix](../reports/sgaf_external_validation_candidate_matrix.md)
- [SGAF External Validation Runbook](../reports/sgaf_external_validation_runbook.md)
- [SGAF Package Handoff Checklist](../reports/sgaf_package_handoff.md)
- [SGAF External Candidate Acquisition Audit](../reports/tables/table_sgaf_external_candidate_acquisition_audit.md)
- [SGAF External Dataset Choice Audit](../reports/tables/table_sgaf_external_dataset_choice_audit.md)
- [SGAF External Validation Manifest Audit](../reports/tables/table_sgaf_external_validation_manifest_audit.md)
- [SGAF External Data Readiness Audit](../reports/tables/table_sgaf_external_data_readiness_audit.md)
- [SGAF External Run Readiness Audit](../reports/tables/table_sgaf_external_run_readiness_audit.md)
- [SGAF External Validation Result Gates](../reports/tables/table_sgaf_external_validation_result_gates.md)
- [SGAF Frozen Protocol Readiness Audit](../reports/tables/table_sgaf_frozen_protocol_readiness.md)
- [SGAF Claim Language Audit](../reports/tables/table_sgaf_claim_language_audit.md)
- [SGAF Package Readiness Audit](../reports/tables/table_sgaf_package_readiness.md)

## Entities
- [Qwen2.5-0.5B-Instruct](entities/Qwen2.5-0.5B-Instruct.md) — Small LLM for routing and verification
- [Oracle Router](entities/Oracle-Router.md) — Upper-bound routing reference
- [Al-Joofi et al.](entities/Al-Joofi-et-al.md) — Related work (nDCG@10 paper)

## Other Concepts
- [SciNCL](concepts/SciNCL.md) — Old scientific dense embedding (superseded by BGE)
- [QLoRA](concepts/QLoRA.md) — Efficient LLM fine-tuning
- [Confidence-Gated Fallback](concepts/Confidence-Gated-Fallback.md) — Calibrated LLM routing fallback
- [Distribution Shift](concepts/Distribution-Shift.md) — Train/test oracle label mismatch
- [NFCorpus](concepts/NFCorpus.md) — Cross-dataset validation (biomedical)
- [BEIR](concepts/BEIR.md) — Heterogeneous retrieval benchmark
- [MiniLM](concepts/MiniLM.md) — Lightweight transformer for cross-encoding
- [SPECTER](concepts/SPECTER.md) — Failed dense retriever ablation

## Syntheses
