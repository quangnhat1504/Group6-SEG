# Wiki Log

Append-only chronological record of every operation.

## [2026-07-11] push-performance | Phase 0-7 full implementation

**Implementation:**
- Added `scripts/audit_split_leakage.py` — 4 leak checks with `--fail-on-leak`
- Added `scripts/run_bge_small_clean_sweep.py` — 6-config Phase 1+2 baseline sweep
- Added `scripts/run_bge_small_phase3_sweep.py` — 9-config triple quality sweep
- Added `scripts/run_bge_small_phase4_sweep.py` — 4-config curriculum sweep
- Added `scripts/tune_rrf_scifact.py` — 20-config RRF k/weight tuning
- Added `scripts/eval_cross_dataset.py` — 4-dataset cross-eval
- Added `scripts/finalize_best_model.py` — 5-step finalize protocol
- Modified `scripts/train_bge_small_scifact.py` — added all flags (exclude IDs, eval-split, epochs/lr/warmup/batch, triple source, negatives/positive, disagreement top-k, hard-negative strategy, curriculum 2-stage, dedup)
- Modified `src/seg_retrieval/fusion.py` — added `weighted_rrf()`

**Wiki update:**
- Created `wiki/sources/push-performance-phase0-3.md` — full Phase 0-7 docs
- Updated `wiki/concepts/Fine-Tune-BGE-RRF.md` — added leak audit + triple knobs sections
- Updated `wiki/overview.md` — Phase 0-7 mention, updated Next Steps
- Updated `wiki/plan.md` — fixed stale script refs, added push-performance link
- Updated `wiki/index.md` — added new source, fixed 6 broken links, added 3 missing concept links
- Updated `wiki/concepts/vstash.md` — bumped date
- Bumped `last_updated` on 14 core concept/entity pages to 2026-07-11
- Lint pass: fixed all broken links, resolved 9 orphans, bumped stale dates

**Scripts count:** 9 new/modified files, 0 new dependencies, all compile OK, dry-run verified

## [2026-06-29] ingest | SEG Research Task List
Initial ingest: created source page for tasks.md.

## [2026-06-29] ingest | Phase 1-2 Progress Report
Initial ingest: created source page for phase1-phase2-report.md.

## [2026-06-29] ingest | SEG Research Paper — Current Progress
Initial ingest: created source page for seg-research-paper.md.

## [2026-06-29] ingest | SEG Research Directions
Initial ingest: created source page for seg-research-directions.md.

## [2026-06-29] ingest | SEG Deep Research Plan
Initial ingest: created source page for seg-research-deep-plan.md.

## [2026-06-29] ingest | Phase 3 Progress — 2026-06-20
Initial ingest: created source page for phase3-progress-2026-06-20.md.

## [2026-06-29] ingest | Phase 3 Progress — 2026-06-24 (Direction 3 Complete)
Initial ingest: created source page for phase3-progress-2026-06-24.md.

## [2026-06-29] ingest | SEG Validation Experiments Report
Initial ingest: created source page for validation-experiments.md.

## [2026-06-29] ingest | Paper — LaTeX Build Instructions
Initial ingest: created source page for paper-readme.md.

## [2026-06-29] ingest | Batch entity and concept page creation
Created 23 entity and concept pages covering all key SEG project topics: SciFact, BM25 Retrieval, Dense Retrieval, Reciprocal Rank Fusion, Query Routing, Selective Reranking, Conformal Risk Control, Query Performance Prediction, Downstream-Utility Reranking, SciNCL, Qwen2.5-0.5B-Instruct, QLoRA, CrossEncoder, Always-Rerank, Confidence-Gated Fallback, Distribution Shift, Semantic Entropy, NFCorpus, BEIR, MiniLM, SPECTER, Oracle Router, Al-Joofi et al.

## [2026-06-29] initial | Wiki agent setup and initial ingest
Wiki agent cloned from llm-wiki-agent, integrated into SEG project. Created 36 wiki pages from 9 source documents, 20 concept pages, 3 entity pages. Built knowledge graph (33 nodes, 66 edges). Health check: all green.

## [2026-06-29] experiment-session | Full pipeline redesign and multi-dataset validation
Ran 20+ experiment configurations across SciFact, NFCorpus, and FiQA. Key finding: BGE-base-en-v1.5 (0.7376) replaces entire old multi-stage pipeline (0.6939). Adaptive RRF pushes to 0.7514. Cross-encoder reranking confirmed harmful on all datasets. Full report in raw/seg-final-report.md.

## [2026-06-29] graph | Knowledge graph rebuilt

33 nodes, 66 edges (66 extracted, 0 inferred).

## [2026-06-30] cleanup | MAP@10, significance, 150-query removal, data/source audit, stale artifact deletion
**Paper audit & fix:**
- MAP@10 computed for all methods across 3 datasets via `scripts/run_significance_tests.py`
- Paired bootstrap significance tests (10K resamples, 95% CI) for all pairwise comparisons
- BGE-base vs SciNCL: significant on all datasets (p<0.001, +0.1842/+0.1228/+0.2601 MAP@10)
- Adaptive RRF vs BGE-base: NOT significant anywhere (p≥0.083)
- CE on ARF: significantly *degrades* on SciFact (p=0.004), neutral elsewhere
- All 13 paper .tex files updated: MAP@10 as primary metric, significance markers, consistent numbers

**Data split rectification:**
- Removed all 150-query evaluation artifacts (was SciFact test split 150/150 calibration/eval)
- All tables now evaluate on full 300 test queries — consistent across all rows
- Old leaky QLoRA training (809 queries incl. dev) → proper trainfit-only (648 queries, dev=161 excluded)
- QLoRA proper nDCG@10 = 0.6359 (old leaky: 0.6270, old calibrate-on-train-dev: 0.5918)

**Codebase cleanup:**
- Deleted 19 stale scripts (Category C: one-off hacks, superseded orchestrators)
- Deleted 8 stale helper scripts (thin wrappers, ablated functionality)
- Deleted 12 stale run files (150-query artifacts, leaky predictions, old split data)
- Deleted 7 stale report files (old paper drafts, validation summaries, research directions)
- Final script count: 27 (down from ~45), all having clear purpose

**Wiki update:**
- `overview.md` — Replaced SciNCL-era thesis with BGE + MAP@10 + significance findings
- `index.md` — Reorganized into Core Concepts (BGE, Adaptive RRF, MAP@10) vs Historical (SciNCL-era)
- Added new concept pages: BGE, Adaptive-RRF, MAP@10, Paired-Bootstrap
- Updated stale concept pages: Conformal-Risk-Control, Selective-Reranking, CrossEncoder, QPP, RRF

## [2026-07-01] paper-fixes | Table audit, SciDocs, performance table, multi-corpus fine-tune

**Paper fixes:**
- Double-checked all 11 sections + 8 tables for number consistency
- Fixed Table 2 caption (`\captio}` → correct LaTeX)
- Fixed Appendix QE numbers (0.6685 → 0.6571)
- Fixed CRC coverage (61% → 14% on full-300)
- Updated all "3 dataset" → "4 dataset" references (9 locations)
- Added SciDocs analysis to validation_experiments.tex
- Updated abstract SciDocs p-value ($p<0.001$ → $p<0.01$)
- Fixed Table 1 BM25+BGE RRF numbers (was BGE-small, now BGE-base 0.7185)
- Regenerated all CSVs for consistency between Table 1 & Table 2
- CE on BGE RRF now = 0.7023 (-0.0162, harmful — consistent pattern)
- Deleted validation_experiments.tex duplicate table (replaced with ref to Table 5)

**Performance table (experiments.tex):**
- Added full Experimental Setup table with: Docs, n_queries, BM25 Setup (s), BGE Setup (s), Retrieval Latency (ms/q), CPU Usage (%), GPU Usage (%)
- All 17 rows × 3 datasets (SciFact, NFCorpus, SciDocs)
- Real measured numbers: BM25 ~10ms/q, BGE encode 30-80s, CE ~27-47ms/q, GPU 88-90%
- Updated `reports/tables/table_cost_comparison.csv` from old numbers (333ms/q SciNCL) to current (10.8ms/q rank_bm25)

**Pipeline diagrams (drawio):**
- Created 4 phase diagrams: Phase 1 (base retrieval), Phase 2 (router), Phase 3-4 (selective rerank), Current (adaptive RRF)
- `raw/pipeline_current.drawio.xml` — current pipeline: BM25+BGE → Adaptive RRF
- `raw/pipeline_phase3-4.drawio.xml` — selective reranking (QPP + CRC)
- `raw/pipeline_phase2.drawio.xml` — router (QLoRA Qwen 0.5B)
- `raw/pipeline_phase1.drawio.xml` — base retrievers

**Multi-corpus fine-tune attempt:**
- Researched vstash recipe: RRF disagreement signal + MNRL loss
- Attempted BGE-base fine-tune 3×: all failed with catastrophic forgetting
- Attempted BGE-small multi-corpus (SciFact + NFCorpus + FiQA + SciDocs): 44K+ triples from SciFact alone, but timeout during FiQA mining
- Created standalone script: `scripts/train_bge_small_scifact.py` — pure SciFact, target 0.75+ NDCG

**Wiki updates:**
- Created: SciDocs.md, FiQA.md, Fine-Tune-BGE-RRF.md, Catastrophic-Forgetting.md
- Updated: index.md (added new concept links), vstash.md (cross-dataset results + own model)
- Logged: this entry

## [2026-07-01] plan | Next steps + wiki updates

- Created `wiki/plan.md` — detailed roadmap (Phases A-D) với priority matrix
- Updated `overview.md`: added fine-tuned model to results table (0.7909), "Our Own Model" marked completed, Pipeline step 4 updated, Next Steps section added, Central Finding updated
- Updated `index.md`: added plan link, marked Fine-Tune as successful
- Updated `concepts/Fine-Tune-BGE-RRF.md`: restructured with success section, full metrics, next steps
- Updated `concepts/vstash.md`: our model now beats vstash, updated hypothesis
- Logged this entry

## [2026-07-01] wiki-ingest | Knowledge ingestion from session
**Wiki pages created/updated:**
- Created: SciDocs.md (concept), FiQA.md (concept), Fine-Tune-BGE-RRF.md (concept), Catastrophic-Forgetting.md (concept)
- Updated: overview.md (added SciDocs column, vstash reference, our own model section, 4 datasets), index.md (added new concept links), vstash.md (cross-dataset + own model info)
- Logged: this entry
**Critical fix:**
- Switched primary metric from MAP@10 to **nDCG@10** (per Al-Joofi 2025 paper, which explicitly states nDCG@10 is the primary comparison measure)
- Replaced paired bootstrap (10K resamples) with **paired t-test + Shapiro-Wilk normality check** (Wilcoxon fallback if normality violated)
- Rewrote `scripts/run_significance_tests.py` — now computes nDCG@10 (primary) + MAP@10 (secondary) for 6 pairs × 3 datasets
- Added `per_query_map()` to `src/seg_retrieval/metrics.py`
- Updated all 13 paper .tex files: tables, sections, captions
- Updated wiki: MAP@10, Al-Joofi-et-al, Paired-Bootstrap (→ Paired T-Test), Overview, BGE, Index
- Created new wiki page: concepts/nDCG@10.md
- Generated new `runs/_sig_tests.json` with fresh numbers

**New significance results (nDCG@10, paired t-test / Wilcoxon):**
- BGE-base vs SciNCL: +0.1737/+0.1444/+0.3110 (p<0.001 on all 3 datasets)
- Adaptive RRF vs BGE-base: not significant (p=0.139/0.688/0.484)
- CE on ARF vs ARF: degrades on SciFact (-0.0374, p=0.015), neutral elsewhere


## [2026-07-11] strategy-shift | Specialist-Generalist Adaptive Fusion

- Shifted next research direction away from multi-corpus labeled-only as the primary novelty claim.
- Added `concepts/Specialist-Generalist-Adaptive-Fusion.md`: query-adaptive fusion of SciFact specialist, generalist dense retrievers, and BM25.
- Updated `overview.md`: current best SciFact specialist = 0.8188 nDCG@10, with weaker cross-dataset transfer.
- Rewrote `plan.md`: Phase 1 static ensemble, Phase 2 query-adaptive fusion, Phase 3 frozen test, Phase 4 failure-driven fine-tuning.
- Updated `Fine-Tune-BGE-RRF.md`, `vstash.md`, and QPP wording to avoid stale multi-corpus/"no leakage" claims.
