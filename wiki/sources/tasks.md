---
title: "SEG Research Task List"
type: source
tags: [project-management, task-tracking]
date: 2026-06-29
source_file: raw/tasks.md
---

## Summary
The SEG research task list tracks all phases of the project from setup through Phase 4. It covers the full pipeline: base retrieval (BM25, Dense, Hybrid RRF), query routing (Random/Majority/Oracle/Classical/QLoRA LLM routers), uncertainty-driven selective reranking, QPP signals, conformal risk control, and downstream-utility reranking. All major milestones through Phase 4 H1/H2 are complete, with H3 (downstream-utility) and cross-dataset validation remaining.

## Key Claims
- Phase 0-3 are complete, Phase 4 H1/H2 (QPP + Conformal) are done
- All core report tables (Tables 1-4) have been drafted
- Five research questions (RQ1-RQ5) guide the investigation
- A Google Sheet tracks all experiment results
- Phase 4 remaining: H3 downstream-utility reranking, cross-dataset validation (NFCorpus/TREC-COVID)

## Connections
- [[SciFact]] — primary evaluation dataset
- [[BM25 Retrieval]] — lexical baseline
- [[Dense Retrieval]] — semantic baseline with [[SciNCL]]
- [[Reciprocal Rank Fusion]] — combines BM25 + Dense
- [[Query Routing]] — per-query retrieval method selection
- [[Selective Reranking]] — uncertainty-gated cross-encoder invocation
- [[Conformal Risk Control]] — principled threshold selection
- [[Query Performance Prediction]] — unsupervised signal for triggering decisions
- [[Downstream-Utility Reranking]] — Phase 4 H3 direction
- [[Qwen2.5-0.5B-Instruct]] — small LLM used for routing
- [[QLoRA]] — efficient fine-tuning method
- [[BEIR]] — evaluation benchmark framework
- [[CrossEncoder]] — reranker model
- [[MiniLM]] — lightweight transformer for cross-encoding
