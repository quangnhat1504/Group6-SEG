# Catastrophic Forgetting in Embedding Fine-Tuning

- **Problem:** Fine-tuning BGE-base (109M) with MNRL on ~2K triples causes NDCG to drop instead of improve. Observed across 3 independent experiments.
- **Observed drops:**
  - V1 (3 epochs, full model, LR=2e-5): −0.0135 (0.7512 → 0.7377)
  - Iterative (5 rounds re-mining, LR=2e-5): −0.0389 (0.7512 → 0.7123)
  - V2 (freeze 6/12 layers, LR=5e-6): −0.0135 (0.7512 → 0.7377)
- **Root cause:** MNRL + in-batch negatives are too aggressive for a model that was heavily pretrained on MS MARCO + NLI. Small fine-tuning dataset pushes embeddings away from original distribution.
- **Fix strategies tried:**
  - Freeze lower 6 layers → same degradation
  - Lower LR (5e-6 → 2x lower than default) → same degradation
  - Early stopping → saves before degradation but no improvement captured
- **What works:** Smaller model (33M BGE-small) with same recipe is stable. Original vstash recipe was designed for 33M.
- **Path forward:**
  - Train BGE-small first (proven), then attempt BGE-base with: freeze 10/12 layers, LR=1e-7, MultiNeg+CrossEntropy hybrid loss, 60K+ triples from multi-corpus mining
  - Or use vstash CLI directly (which includes tuned hyperparameters for eval-gating)
- **Date added:** 2026-07-01