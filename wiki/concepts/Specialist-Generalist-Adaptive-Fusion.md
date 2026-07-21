---
title: "Specialist-Generalist Adaptive Fusion"
type: concept
tags: [retrieval, adaptive-fusion, qpp, performance]
last_updated: 2026-07-20
---

# Specialist-Generalist Adaptive Fusion

## Summary
Specialist-Generalist Adaptive Fusion is the new primary performance direction for SEG. Instead of training one more dense model and hoping it generalizes, the method treats the current SciFact fine-tuned BGE-small as a **specialist**, BGE-base/vstash as **generalists**, and BM25 as a lexical anchor. The system chooses or weights these retrievers per query using label-free agreement and QPP features.

## Motivation
The current best local model, `runs/finetuned/bge-small-final`, is very strong on SciFact but weaker cross-dataset:

| Model | SciFact | NFCorpus | FiQA | SciDocs |
|-------|---------|----------|------|---------|
| BGE-small final | **0.8188** | 0.3505 | 0.3635 | 0.1893 |
| vstash rrf-v3 | 0.7707 | 0.3667 | **0.4825** | **0.2150** |
| BGE-base | 0.7376 | **0.3695** | 0.3909 | 0.2147 |

This shows a specialization/generalization tradeoff. Fine-tuning makes a strong SciFact specialist, but hurts transfer. Adaptive fusion can preserve the specialist's SciFact gains while borrowing robustness from generalist retrievers.

## Current Dev Evidence

Static global weighting has now been tested on SciFact dev.

| Method | Dev nDCG@10 | Delta vs BGE-small | Interpretation |
|---|---:|---:|---|
| BGE-small-final | 0.9052 | +0.0000 | specialist baseline |
| BGE-base | 0.7234 | -0.1818 | weaker global dev retriever |
| Best static weighted RRF | 0.9053 | +0.0000 | no meaningful gain; best uses BGE-base weight 0 |
| Oracle over BM25/BGE-small/BGE-base | 0.9235 | +0.0182 | query-adaptive headroom exists |

BGE-base strictly beats BGE-small on `10` dev queries. Static RRF cannot use that subset without diluting the specialist, so the next contribution must be query-adaptive rather than global.

## Query-Adaptive Result

Source: `reports/tables/table_query_adaptive_fusion.md`

Train on `trainfit`, validate on `dev`, and evaluate on `test` with BM25, BGE-small-final, and BGE-base components.

| Stage | Method | Dev nDCG@10 | Dev Delta | Test nDCG@10 | Test Delta | Test Switch |
|---|---|---:|---:|---:|---:|---:|
| C:bge_small | Specialist baseline | 0.9052 | +0.0000 | 0.8188 | +0.0000 | N/A |
| O1 | Oracle component router | 0.9235 | +0.0182 | 0.8786 | +0.0598 | N/A |
| A1 | Multiclass adaptive router | 0.8655 | -0.0397 | 0.7827 | -0.0361 | 0.5833 |
| A2 | Binary BGE-base rescue gate | 0.9117 | +0.0065 | 0.8015 | -0.0173 | 0.3500 |
| A3 | Coverage-controlled BGE-base rescue gate | 0.9098 | +0.0046 | 0.8216 | +0.0028 | 0.0500 |
| A4 | Coverage-controlled BM25 lexical rescue gate | 0.9052 | +0.0000 | 0.8188 | +0.0000 | 0.0000 |
| A5 | Dual BGE-base + BM25 rescue gate | 0.9098 | +0.0046 | 0.8216 | +0.0028 | 0.0500 |

Interpretation:

- A1 fails because multiclass routing over-switches to the weaker generalist.
- A2 passes the dev success criterion but fails on test because an absolute probability threshold does not transfer.
- A3 is the current best candidate because coverage control keeps generalist use bounded and stays positive on both dev and test.
- A4/A5 are useful negative ablations: under clean trainfit selection, BM25 rescue coverage is `0` because trainfit has no BM25 oracle-positive labels.
- Paired bootstrap does not support a significance claim for A3 yet: dev delta `+0.004629`, CI `[-0.003003, +0.013754]`, p=`0.2530`; test delta `+0.002813`, CI `[-0.003648, +0.010259]`, p=`0.4304`.

## Frozen Robustness Result

Source: `reports/tables/table_frozen_sgaf_robustness.md`

The frozen A3 recipe (`C=0.1`, coverage `0.05`) was trained on SciFact `trainfit` and applied without target-label tuning.

| Dataset | BGE-small | BGE-base | Oracle | Frozen A3 | A3 Delta | Significant |
|---|---:|---:|---:|---:|---:|---|
| SciFact | 0.8188 | 0.7376 | 0.8786 | 0.8216 | +0.0028 | no |
| NFCorpus | 0.3505 | 0.3695 | 0.4249 | 0.3530 | +0.0025 | yes, p=0.0488 |
| FiQA | 0.3635 | 0.3909 | 0.4650 | 0.3653 | +0.0017 | no |
| SciDocs | 0.1893 | 0.2147 | 0.2666 | 0.1902 | +0.0008 | no |

Duplicate-filtered SciFact keeps the A3 delta: BGE-small-final drops from `0.8188` to `0.8176`, while frozen A3 drops from `0.8216` to `0.8204`.

Interpretation:

- Fixed 5% coverage is robust: it avoids the over-switching failure of A1/A2 and does not collapse on transfer datasets.
- It is also too conservative: BGE-base is globally stronger than BGE-small on NFCorpus, FiQA, and SciDocs, but frozen A3 still routes only 5% of queries to BGE-base.
- The next improvement should therefore be adaptive coverage control, not a return to BM25 rescue.

## Adaptive Coverage Result

Source: `reports/tables/table_adaptive_coverage_sgaf.md`

This phase freezes the BGE-base rescue ranking model from SciFact `trainfit` and changes only the coverage budget. The deployable idea is label-free source/uncertainty shift coverage; the oracle sweep is diagnostic only.

| Dataset | Fixed A3 | Source-shift | Uncertainty-shift | Conservative-shift | Oracle coverage sweep |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.8216 | 0.8218 | 0.8218 | 0.8225 | 0.8216 |
| NFCorpus | 0.3530 | 0.3578 | 0.3559 | 0.3554 | 0.3695 |
| FiQA | 0.3653 | 0.3718 | 0.3774 | 0.3681 | 0.3909 |
| SciDocs | 0.1902 | 0.1951 | 0.1962 | 0.1916 | 0.2147 |

Significance versus fixed A3:

- FiQA source-shift: `+0.0065`, p=`0.0016`; uncertainty-shift: `+0.0121`, p=`0.0010`.
- SciDocs source-shift: `+0.0049`, p=`0.0020`; uncertainty-shift: `+0.0060`, p=`0.0004`.
- SciFact remains stable; NFCorpus improves directionally but is not significant.

Interpretation:

- Coverage control is a stronger candidate than fixed 5% A3 for cross-dataset transfer.
- The ranker and components stay fixed, so the contribution is isolated to budget selection.
- Because this formula was developed during the phase, it should be frozen and re-evaluated before being treated as a final result.

Failure analysis shows why adaptive coverage helps:

| Dataset | Fixed captured BGE-base rescues | Source-shift captured | Uncertainty-shift captured | BGE-base rescue headroom |
|---|---:|---:|---:|---:|
| SciFact | 4 | 7 | 7 | 35 |
| NFCorpus | 7 | 51 | 27 | 117 |
| FiQA | 18 | 39 | 61 | 208 |
| SciDocs | 16 | 67 | 87 | 364 |

Recommendation at the end of Phase 6:

- Freeze **uncertainty-shift adaptive coverage** as the performance candidate.
- Keep **source-shift adaptive coverage** as a safer/interpretability ablation.
- Do not re-enable BM25 rescue unless a clean split produces BM25-positive labels.

This recommendation was superseded by Phase 7 BGE-base mode switching.

## Adaptive Coverage Predecessor

Source: `reports/tables/table_final_sgaf_synthesis.md`

Previous candidate: **Uncertainty-shift Adaptive Coverage SGAF**.

Formula:

`coverage = clamp(0.05 + 0.08*max(0,-z(bge_small_gap)) + 0.04*max(0,-z(bge_small_std10)), 0.02, 0.40)`

| Dataset | BGE-small | Fixed A3 | Final adaptive | Delta vs BGE-small | Delta vs Fixed A3 |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.8188 | 0.8216 | 0.8218 | +0.0030 | +0.0001 |
| NFCorpus | 0.3505 | 0.3530 | 0.3559 | +0.0055 | +0.0030 |
| FiQA | 0.3635 | 0.3653 | 0.3774 | +0.0138 | +0.0121 |
| SciDocs | 0.1893 | 0.1902 | 0.1962 | +0.0068 | +0.0060 |

Overall delta vs BGE-small is `+0.0073` average across the four datasets, and overall delta vs fixed A3 is `+0.0053`. This predecessor is positive on every dataset and significant over fixed A3 on FiQA and SciDocs.

Caveat: it remains useful as an ablation, but Frozen B5 is now the stronger current candidate.

## BGE-base Comparison

Changing the comparison baseline from BGE-small to BGE-base exposes the next performance bottleneck. Current adaptive SGAF beats BGE-base on SciFact by `+0.0842`, but loses on NFCorpus, FiQA, and SciDocs by about `0.0136`, `0.0136`, and `0.0185` nDCG@10 respectively. Transfer-only average is therefore still lower than BGE-base.

The interpretation is:

- current SGAF is a specialist-preserving adaptive method, not a universal BGE-base replacement;
- transfer gap is mostly a coverage-controller issue, because oracle coverage sweep selects `1.0` for NFCorpus, FiQA, and SciDocs;
- the next phase should test a label-free BGE-base mode switch before adding any expensive reranker.

Detailed plan: `reports/sgaf_bge_base_mode_switch_plan.md`.

## Phase 7 B4 Diagnostic

Source: `reports/tables/table_sgaf_mode_switch_ablation.md`

B4 tested whether the BGE-base transfer gap is caused by the max coverage cap. The answer is no: increasing the cap from `0.40` to `1.00` while keeping the current uncertainty formula leaves all coverages and scores unchanged.

| Method | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|
| Current uncertainty coverage | 0.3098 | -0.0152 | +0.0030 |
| Cap-only max 1.00 | 0.3098 | -0.0152 | +0.0030 |
| Gain 4.0 max 1.00 | 0.3176 | -0.0074 | -0.0097 |
| Gain 8.0 max 1.00 | 0.3251 | +0.0000 | -0.0103 |

Interpretation:

- The current formula is too conservative before it hits the cap.
- Global gain can recover transfer but damages SciFact, so it is not the final method.
- The next useful controller is a batch-level source-shift mode that enables high BGE-base coverage only on shifted batches.

B5 batch mode confirms this direction:

| Method | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|
| Current uncertainty coverage | 0.3098 | -0.0152 | +0.0030 |
| Batch shift t2.0 gain 6.0 | 0.3249 | -0.0001 | +0.0030 |
| Batch shift t2.0 gain 8.0 | 0.3251 | +0.0000 | +0.0030 |

This is the first candidate that keeps the SciFact specialist gain while essentially recovering BGE-base transfer. It should be treated as an exploratory Phase 7 candidate until one B5 recipe is frozen and re-evaluated.

## Frozen B5 Candidate

Source: `reports/tables/table_final_sgaf_mode_switch.md`

Frozen recipe:

- batch shift threshold `2.0`;
- shifted uncertainty gain `6.0`;
- shifted cap `1.0`;
- SciFact trainfit BGE-base rescue classifier, `C=0.1`.

| Method | Avg nDCG@10 | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|---:|
| BGE-small specialist | 0.4305 | 0.3011 | -0.0239 | +0.0000 |
| BGE-base generalist | 0.4282 | 0.3251 | +0.0000 | -0.0812 |
| Current adaptive SGAF | 0.4378 | 0.3098 | -0.0152 | +0.0030 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | -0.0001 | +0.0030 |

Duplicate-filtered SciFact remains stable: current adaptive and frozen B5 both score `0.8206` nDCG@10 after excluding the two known duplicate test queries.

The research framing should now be:

> A frozen batch-level mode switch preserves specialist-first behavior on source-like batches and escalates to generalist coverage under cheap distribution-shift evidence, recovering BGE-base transfer without sacrificing SciFact.

Remaining caveat: threshold and gain were discovered during Phase 7 exploration, so future validation must keep them fixed.

## Core Method
1. Retrieve candidates from multiple retrievers: BM25, BGE-base, SciFact-specialist BGE-small, and optionally vstash.
2. Build a deduplicated candidate union.
3. Fuse rankings with weighted RRF or score-normalized CombSUM.
4. Predict per-query weights from agreement/QPP features.
5. Freeze weights or gating rules on SciFact dev; evaluate final test once.

## Candidate Features
- BM25 top score and score gap
- Dense top score and score entropy
- Rank overlap between BM25 and dense retrievers
- RRF max score
- Agreement between BGE-base, BGE-small-final, and vstash
- Query length and IDF statistics
- Specialist-vs-generalist disagreement indicators

## Research Claim
A clean claim is not "we trained another model". The claim is:

> Fine-tuning creates a high-performing specialist but weakens transfer. A leakage-aware, query-adaptive fusion layer combines specialist and generalist retrievers per query, aiming to improve SciFact while preserving cross-domain robustness.

Candidate claim:

> Adaptive coverage SGAF separates query ranking from budget selection: a source-trained rescue ranker orders queries by generalist-rescue likelihood, while a cheap source-shift/uncertainty controller adjusts how much generalist coverage is allowed under domain shift.

## Protocol
- Use SciFact `trainfit` for learned gates and feature models.
- Use SciFact `dev` for selecting fusion weights and gates.
- Freeze the recipe before final SciFact `test`.
- Report SciFact duplicate-filtered sensitivity.
- Evaluate FiQA, NFCorpus, and SciDocs as zero-shot transfer only.

## Implementation Roadmap
1. `tune_static_ensemble.py`: weighted RRF over existing run CSVs. Completed on dev; no meaningful gain.
2. `train_query_adaptive_fusion.py`: train a lightweight gate/weight predictor on trainfit/dev. Completed for A1/A2/A3 plus A4/A5 BM25-rescue checks.
3. Bootstrap/significance for A3 vs BGE-small-final. Completed; positive deltas are not significant on dev or test.
4. Add a separate low-coverage BM25 rescue gate because test oracle still has BM25 wins. Completed; clean trainfit selection chooses BM25 coverage `0`.
5. Evaluate cross-dataset transfer after the SciFact recipe is frozen. Completed; A3 is directionally positive on all four datasets, significant only on NFCorpus.
6. Adaptive coverage control. Exploratory ablation completed; strongest transfer gains appear on FiQA and SciDocs.
7. Adaptive coverage failure analysis. Completed; uncertainty-shift captures the most BGE-base rescues on FiQA/SciDocs.
8. Final SGAF synthesis. Superseded by Phase 7; uncertainty-shift adaptive coverage is now the predecessor ablation.
9. BGE-base mode-switch ablation. Completed; cap-only is a no-op, global gain hurts SciFact, batch shift mode preserves SciFact and recovers transfer.
10. Frozen B5 mode-switch core. Completed; `threshold=2.0`, `gain=6.0`, shifted cap `1.0`.
11. Frozen P3 rank-window smoothing. Completed; `window=20`, `alpha=0.10`, current strongest experimental candidate.
12. P4 residual specialist fallback. Completed as optional diagnostic; positive but too small to replace P3 without external validation.
13. Failure-driven mining or new frozen validation batch: future extension.

## Risks
- SciFact dev is small, so keep grids small and report test results as frozen-protocol, not leaderboard tuning.
- Cross-dataset numbers must not be used for model selection.
- vstash is an external model; keep results with and without it if authorship matters.
- B5 threshold/gain, P3 window/alpha, and optional P4 feature/fraction were selected during exploration; future validation must keep them fixed.
