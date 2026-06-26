# SEG Phase 3 Progress Report - 2026-06-24

Direction 3 (downstream-utility reranking) is COMPLETE and is a NEGATIVE RESULT, now written up in report Section 6.10.

Method: for each (claim, abstract) pair, read Qwen2.5-0.5B-Instruct's label distribution over SUPPORT/REFUTE/NEI from the log-probabilities of those label strings in one forward pass (discrete sampling was tried first but collapses to a constant NEI on a 0.5B model). Three label-free utility signals: verification confidence, semantic entropy (Farquhar et al. 2024), and an InfoGain-RAG-style information gain vs a claim-only prompt.

Key numbers (SciFact test, 300 queries, top-20):
- Signal separation of relevant vs non-relevant abstracts is weak: AUC ~0.60-0.64. The 0.5B verifier is overconfident (non-relevant mean confidence 0.924).
- Reranking never beats the Hybrid base (nDCG@10 0.6583): pure reorder 0.351; best blend +0.0006 (noise); conformal-gated selective application lowers nDCG@10 at every coverage (e.g. 0.5396 at 61% coverage).
- For comparison the cross-encoder Always-Rerank reaches ~0.728.

Mechanism: SciFact has ~1 relevant abstract per claim hidden among 20 candidates; a small generative verifier cannot surface it the way a purpose-built cross-encoder can. This strengthens the H1/H2 thesis: value is in cheap label-free signals deciding WHEN to invoke a strong reranker, not in making the expensive stage reason about the task.

Infra: GPU enabled (RTX 5070 Ti, cu128 PyTorch); 0.5B inference ~37 ms/pair vs ~870 ms on CPU. New code: src/seg_retrieval/utility_rerank.py, scripts/run_utility_rerank.py, scripts/analyze_utility_rerank.py.

Next (optional): retry with a 1.5B/3B verifier; validate H1/H2 on a 2nd BEIR dataset.
