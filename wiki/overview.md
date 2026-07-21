---
title: "SEG Project Wiki"
type: synthesis
tags: [overview, meta]
last_updated: 2026-07-20
---

## Summary

The SEG project is a student-scale research codebase for **scientific document retrieval**. It first shows that a modern dense retriever (BGE-base-en-v1.5, 109M) is a much stronger baseline than the old SciNCL + RRF + cross-encoder stack. The current improvement direction is narrower and more defensible: **Specialist-Generalist Adaptive Fusion**, especially the frozen B5 mode switch that preserves the SciFact specialist while escalating to BGE-base on shifted batches. All results use BEIR-style datasets with nDCG@10 as the primary metric per [[Al-Joofi-et-al]].

**Our current strongest experimental candidate is Frozen P3 Rank-Window Smoothing SGAF.** It builds on Frozen B5, keeps SciFact at 0.8218 nDCG@10, and raises transfer average to 0.3293. Frozen B5 remains the cleaner mode-switch core; P3 is stronger on current evidence but still needs external frozen validation. A P4 residual fallback diagnostic raises transfer average to 0.3314, but the gain is small and should remain optional until external validation.

## Central Finding
> Fine-tuning creates a very strong SciFact specialist, but also exposes a specialization/generalization tradeoff. The strongest current path is a cheap mode-switch policy plus post-retrieval rank-window smoothing: use the SciFact specialist on source-like batches, fall back to BGE-base under distribution shift, then blend a weak specialist prior only inside a small top-rank window.

## Pipeline (Current)
1. **BM25** → Lexical retrieval (nDCG@10=0.6523, cheap, ~10ms/q with rank_bm25)
2. **BGE-base-en-v1.5** → Semantic retrieval (nDCG@10=0.7376) — already beats old SciNCL RRF+CE (0.6939)
3. **Adaptive RRF** → IDF-weighted fusion (nDCG@10=0.7514) — best fusion, 91% of Oracle Router
4. **BGE-small final (SciFact specialist, 33M)** → **0.8188 nDCG@10** on SciFact, but weaker cross-dataset transfer
5. **Frozen B5 Mode-Switch SGAF** → **0.8218** on SciFact and **0.3249 transfer avg**, nearly matching BGE-base transfer without losing the specialist gain
6. **Frozen P3 Rank-Window Smoothing SGAF** → **0.8218** on SciFact and **0.3293 transfer avg**, current strongest experimental candidate
7. **P4 residual specialist fallback** → optional diagnostic, **0.3314 transfer avg**, not yet a main claim
8. **Cross-Encoder** → **HARMFUL** on all BGE pipelines (drops from 0.7514 to 0.7140, p=0.015)

## Key Results (nDCG@10 primary, 300-1,000 test queries)

| Method | SciFact | NFCorpus | FiQA | SciDocs |
|--------|---------|----------|------|---------|
| BM25 | 0.6523 | 0.3079 | 0.2167 | 0.1495 |
| SciNCL (old) | 0.5640 | 0.2252 | 0.0800 | 0.1951 |
| **BGE-base** | 0.7376 | **0.3695** | 0.3909 | **0.2147** |
| Adaptive RRF | 0.7514 | 0.3719 | 0.3827 | 0.2097 |
| CE on ARF | 0.7140 | 0.3674 | 0.3910 | 0.1910 |
| *vstash rrf-v3 (ref)* | *0.7707* | *0.3667* | ***0.4825*** | *0.2150* |
| **BGE-small final (ours)** | **0.8188** | 0.3505 | 0.3635 | 0.1893 |
| **Frozen B5 SGAF (ours)** | **0.8218** | 0.3692 | 0.3909 | **0.2147** |
| **Frozen P3 SGAF (ours)** | **0.8218** | **0.3744** | 0.3960 | **0.2173** |

**Significance (paired t-test, Shapiro-Wilk check):**
- BGE-base vs SciNCL: +0.1737 (p<0.001, ***) / +0.1444 (***) / +0.3110 (***) / +0.0196 (**)
- Adaptive RRF vs BGE-base: not significant on any dataset (p≥0.139)
- CE on ARF vs ARF: degrades on SciFact (−0.0374, p=0.015), harms SciDocs (−0.0187, ***)

## External Reference: vstash (Stffens/bge-small-rrf-v3)
- 33M model fine-tuned with RRF disagreement signal on multi-corpus
- **SciFact: 0.7707** — beats Oracle Router (0.7617), beats BGE-base (0.7376)
- Full metrics: MAP=0.7227, R@10=0.9110, R@100=0.9900, MRR=0.7316
- FiQA: 0.4825 (+0.0916 vs BGE-base), NFCorpus: neutral, SciDocs: neutral
- See: [[vstash]], [[Fine-Tune BGE RRF]]

## Our Own Model (completed ✔)
- **Best SciFact specialist:** BGE-small final → **0.8188 nDCG@10** on SciFact (`runs/cross_dataset_bge-small-final.json`).
- **Tradeoff:** The same model drops on NFCorpus (0.3505), FiQA (0.3635), and SciDocs (0.1893), while vstash/BGE-base are more robust cross-dataset.
- **Leakage caveat:** SciFact source archive contains two duplicate train/test claim-evidence pairs. Current clean runs exclude train IDs `871,1291`; final reporting should include duplicate-filtered sensitivity.
- **Next:** validate Frozen P3 (`window=20`, `alpha=0.10`) and optional P4 (`small_minus_base_top`, `10%`) without retuning on any new held-out batch or dataset before making a stronger generalization claim.
- BGE-base attempts (3×) failed with catastrophic forgetting — see [[Catastrophic Forgetting]].

## Project Structure
- **`src/seg_retrieval/`** — Library: retrievers, fusion, metrics, QPP, rerank, router, oracle, io, config
- **`scripts/`** — Executable experiments
- **`paper/`** — LaTeX paper with nDCG@10, main result tables, significance checks, and appendix ablations
- **`raw/`** — DrawIO pipeline diagrams, experiment reports
- **`data/`** — BEIR datasets (SciFact, NFCorpus, FiQA, SciDocs)
- **`runs/`** — All run files and evaluation artifacts
- **`wiki/`** — This knowledge base

## Next Steps
See full plan at [[plan]]. **Push Performance Phase 0-8 implemented; Frozen P3 is the current strongest experimental candidate.**

**Immediate:**
- Keep Frozen B5 fixed (`threshold=2.0`, `gain=6.0`, `cap=1.0`, `C=0.1`), Frozen P3 fixed (`window=20`, `alpha=0.10`), and optional P4 fixed (`small_minus_base_top`, `10%`) for the next validation.
- Add a clean held-out/future-batch validation before claiming generalization beyond the current evaluated datasets.
- Report the BGE-base comparison as specialist-preserving transfer recovery, not universal outperformance.

**Secondary:**
- Use failure cases from specialist-vs-generalist disagreement for later hard-negative mining.
- Keep multi-corpus labeled-only training as a baseline/ablation, not the main novelty claim.

## Core Concepts
- [[BGE]] — BGE-base-en-v1.5, the modern embedding model
- [[Adaptive RRF]] — Per-query IDF-weighted reciprocal rank fusion
- [[nDCG@10]] — Primary evaluation metric (aligned with Al-Joofi 2025)
- [[Paired-Bootstrap]] — Significance testing methodology
- [[Fine-Tune BGE RRF]] — Self-supervised fine-tuning via disagreement signal — **now completed!**
- [[Catastrophic Forgetting]] — Why fine-tuning BGE-base fails
- [[vstash]] — External reference model (0.7707 NDCG)
- [[SciDocs]] — Citation prediction dataset (BEIR, 25K docs)
- [[FiQA]] — Financial QA dataset (BEIR, 57K docs)
- [[plan]] — Detailed next steps roadmap

