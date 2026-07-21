---
title: "Query Performance Prediction"
type: concept
tags: [unsupervised, pre-retrieval, trigger-signal]
sources: [seg-research-paper, seg-research-directions, seg-research-deep-plan, validation-experiments]
last_updated: 2026-07-11
---

## Summary
QPP predicts retrieval effectiveness before or after retrieval without relevance labels. In SEG, it serves as the trigger signal for selective reranking and potential routing decisions.

## Computed Signals
- Hybrid max score: strongest predictor, Kendall +0.499 vs base nDCG, -0.166 vs gain-from-rerank
- WIG (Weighted Information Gain): based on top-k score divergence
- NQC (Normalized Query Commitment): score dispersion metric
- Score std: standard deviation of top-k scores
- Score gap: top-1 vs top-2 score difference
- Retriever disagreement: BM25-Dense top-10 overlap

## Key Findings
- Hybrid max score predicts gain-from-reranking ~3× better than score-gap heuristic
- All QPP signals weakly correlated with gain (strongest |τ|=0.166)
- hybrid_max is #1 on both train (|τ|=0.184) and test (|τ|=0.166), but test-selected QPP variants should be treated as exploratory; future gates should be selected on SciFact dev only
- QPP signal identity differs across datasets: hybrid_max for SciFact, bm25_std for [[NFCorpus]]
- Learned QPP regression (Direction 1 in deep plan) could improve τ to 0.25-0.35

## Related To
- [[Conformal Risk Control]] — uses QPP signal as nonconformity score
- [[Selective Reranking]] — the gating mechanism
- [[Reciprocal-Rank-Fusion]] — hybrid_max signal source
- [[BM25 Retrieval]] — bm25_std signal source
