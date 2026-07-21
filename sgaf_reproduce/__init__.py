"""SGAF B5+P3 reproduction package.

Copy this entire directory to reproduce the Specialist-Generalist
Adaptive Fusion (SGAF) pipeline described in the SEG paper.

REQUIREMENTS:
  - Python 3.13+
  - GPU with 16 GB VRAM (tested on RTX 5070 Ti)
  - BEIR datasets: SciFact, NFCorpus, FiQA, SciDocs

INSTALL:
  pip install torch sentence-transformers numpy pandas scikit-learn pyyaml

STEP 1 — Prepare data:
  Place BEIR datasets in:
    data/scifact/corpus.jsonl   (5183 docs: _id, title, text)
    data/scifact/queries.jsonl   (1109 queries: _id, text)
    data/nfcorpus/...  (NFCorpus BEIR format)
    data/fiqa/...      (FiQA BEIR format)
    data/scidocs/...   (SciDocs BEIR format)

STEP 2 — Fine-tune BGE-small specialist (Phase 0):
  python scripts/train_bge_small_scifact.py --data-dir data/scifact
  Output: runs/finetuned/bge-small-scifact-rrf/

STEP 3 — Run transfer retrieval (get BGE-small/BGE-base rankings):
  python scripts/run_finetuned_dense_retrieval.py --dataset scifact
  python scripts/run_finetuned_dense_retrieval.py --dataset nfcorpus
  python scripts/run_finetuned_dense_retrieval.py --dataset fiqa
  python scripts/run_finetuned_dense_retrieval.py --dataset scidocs

STEP 4 — B5 mode-switch ablation (Phase 7):
  python scripts/run_sgaf_mode_switch_ablation.py
  Output: runs/fusion/sgaf_mode_switch_ablation/

STEP 5 — P3 rank-window smoothing (Phase 8):
  python scripts/run_sgaf_rank_window_smoothing_ablation.py
  Output: runs/fusion/phase8_rank_window_smoothing/

STEP 6 — Summarize final candidate:
  python scripts/summarize_sgaf_mode_switch_final.py
  python scripts/summarize_sgaf_p3_final_candidate.py
  Output: runs/fusion/final_sgaf_p3_smoothing/

STEP 7 — Benchmark (optional):
  python scripts/benchmark_sgaf_pipeline.py

KEY RESULTS:
  BGE-small specialist       SciFact nDCG@10 = 0.8188, Transfer Avg = 0.3011
  BGE-base generalist        SciFact nDCG@10 = 0.7376, Transfer Avg = 0.3251
  Frozen B5 mode-switch      SciFact nDCG@10 = 0.8218, Transfer Avg = 0.3249
  Frozen P3 rank-window      SciFact nDCG@10 = 0.8218, Transfer Avg = 0.3293

COST:
  Offline: ~45s (two models: BGE-small 33M + BGE-base 109M, 5183 docs)
  Online:  ~18 ms/q (BGE-small enc + BGE-base enc + B5 routing + P3 smoothing)
  B5+P3 CPU overhead: <0.1 ms/q (~0.5% of total)

ARCHITECTURE:
  BGE-small (specialist) ──┐
  BGE-base (generalist) ───┤──> [B5 batch mode-switch] ──> [P3 smoothing] ──> ranking
                            ↑                            ↑
                    5 features extracted           CPU-only, top-20 RRF blend
                    from both rankings             only in fallback batches
"""

import sys
from pathlib import Path

# This file serves as the documentation entry point.
print(__doc__)
print(f"\nPackage location: {Path(__file__).resolve().parent}")
print("\nFor detailed instructions, see README.md in this directory.")
