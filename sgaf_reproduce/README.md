# SGAF B5+P3 Reproduction Package

## What This Is

A self-contained package to reproduce the **Specialist-Generalist Adaptive Fusion (SGAF)** pipeline from the SEG paper. The pipeline combines a fine-tuned BGE-small specialist (33M) with a BGE-base generalist (109M) using a cheap batch-level mode-switch (B5) and optional rank-window smoothing (P3).

## Quick Start

```bash
# 1. Install dependencies
pip install torch sentence-transformers numpy pandas scikit-learn pyyaml

# 2. Place BEIR data in data/ (see Data Format below)

# 3. Fine-tune specialist (Phase 0)
python scripts/train_bge_small_scifact.py --data-dir data/scifact

# 4. Run transfer retrieval
python scripts/run_finetuned_dense_retrieval.py --dataset scifact
python scripts/run_finetuned_dense_retrieval.py --dataset nfcorpus
python scripts/run_finetuned_dense_retrieval.py --dataset fiqa
python scripts/run_finetuned_dense_retrieval.py --dataset scidocs

# 5. B5 mode-switch ablation
python scripts/run_sgaf_mode_switch_ablation.py

# 6. P3 rank-window smoothing
python scripts/run_sgaf_rank_window_smoothing_ablation.py

# 7. Summarize final candidate
python scripts/summarize_sgaf_mode_switch_final.py
python scripts/summarize_sgaf_p3_final_candidate.py
```

## Expected Results (nDCG@10)

| Method | SciFact | NFCorpus | FiQA | SciDocs | Transfer Avg |
|---|---|---|---|---|---|
| BGE-small specialist | 0.8188 | 0.3505 | 0.3635 | 0.1893 | 0.3011 |
| BGE-base generalist | 0.7376 | 0.3695 | 0.3909 | 0.2147 | 0.3251 |
| **Frozen B5 mode-switch** | 0.8218 | 0.3692 | 0.3909 | 0.2147 | 0.3249 |
| **Frozen P3 rank-window** | 0.8218 | 0.3744 | 0.3960 | 0.2173 | **0.3293** |

## Cost (RTX 5070 Ti, SciFact 5183 docs)

| Stage | Offline | Online (per query) | GPU |
|---|---|---|---|
| BGE-small encode (33M) | 8.9s | 8.8 ms/q | Yes |
| BGE-base encode (109M) | 22.0s | 8.9 ms/q | Yes |
| B5 routing (batch) | — | **<0.001 ms/q** | No (CPU) |
| P3 smoothing (batch) | — | **0.02 ms/q** | No (CPU) |
| **Total SGAF** | **45s** | **17.8 ms/q** | 2x encode |

## Data Format

Each dataset needs two JSONL files:

**corpus.jsonl:**
```json
{"_id": "4983", "title": "Microstructural development...", "text": "Abstract text..."}
```

**queries.jsonl:**
```json
{"_id": "1", "text": "0-dimensional biomaterials..."}
```

## File Listing

```
sgaf_reproduce/
├── README.md
├── __init__.py
├── configs/
│   └── scifact.yaml
├── scripts/
│   ├── _bootstrap.py
│   ├── train_bge_small_scifact.py          # Phase 0: Fine-tune specialist
│   ├── run_finetuned_dense_retrieval.py     # Phase 3: Dense retrieval for transfer
│   ├── run_sgaf_mode_switch_ablation.py    # Phase 7: B5 ablation
│   ├── run_sgaf_rank_window_smoothing_ablation.py  # Phase 8: P3 ablation
│   ├── summarize_sgaf_mode_switch_final.py  # Summarize B5 final candidate
│   ├── summarize_sgaf_p3_final_candidate.py # Summarize P3 final candidate
│   ├── evaluate_frozen_sgaf_transfer.py     # Cross-dataset transfer evaluation
│   └── benchmark_sgaf_pipeline.py           # Cost benchmark
├── seg_retrieval/
│   ├── __init__.py
│   ├── config.py
│   ├── datasets.py
│   ├── fusion.py
│   ├── io.py
│   ├── metrics.py
│   ├── oracle.py
│   ├── qpp.py
│   ├── rerank.py
│   ├── retrievers.py
│   ├── router.py
│   ├── types.py
│   ├── uncertainty.py
│   └── utility_rerank.py
└── runs/
    ├── finetuned/bge-small-scifact-rrf/  # Pre-trained specialist configs
    └── scifact/                          # Trainfit retrieval runs
```

## Key Parameters (Frozen)

| Parameter | Value | Description |
|---|---|---|
| Batch shift threshold (tau) | 2.0 | B5 mode-switch trigger |
| Shifted gain | 6.0 | B5 uncertainty gain in fallback |
| Max coverage | 1.0 | B5 max fallback coverage |
| P3 window (w) | 20 | Top-k to smooth |
| P3 alpha | 0.10 | BGE-small prior weight |
| RRF constant (k) | 60 | Fusion parameter |

## Citation

If you use this code, please cite:
```
SEG Team. "Modernizing Multi-Stage Scientific Retrieval: From Strong Dense
Retrieval to Specialist-Generalist Mode Switching." Technical Report, 2026.
```
