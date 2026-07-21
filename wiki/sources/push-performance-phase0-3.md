---
title: "Push Performance Phase 0-7 Implementation"
type: source
tags: [progress-report, phase0-7, leak-audit, triple-quality, curriculum, rrf-tuning, cross-dataset, finalize]
date: 2026-07-11
---

## Summary

Implemented full Push Performance plan (Phase 0-7). Added 9 new scripts, 1 modified library file (fusion.py), zero new dependencies. All compile OK, dry-run verified with `--eval-split dev` properly wired. Test is frozen until finalize.

## Phase 0 — Leak Audit & Clean Protocol

### New files
- `scripts/audit_split_leakage.py` — 4 leak checks (query_id overlap, exact query_text overlap, qrel tuple overlap, same text + same gold doc), `--fail-on-leak` for CI, `--run-csv` for duplicate-filtered metrics
- `scripts/run_bge_small_clean_sweep.py` — 6-config grid sweep (epochs/lr/warmup/batch), guard leak before each run, `--eval-split dev` wired correctly, skip existing results

### Modified files
- `scripts/train_bge_small_scifact.py` — added `--exclude-train-query-ids`, `--output-dir`, `--epochs`, `--batch-size`, `--warmup-ratio`, `--learning-rate`, `--eval-split`, `--seed`; saves `manifest.json` per run

### Audit results
- **Expected leaks found:** train=871 → test=870 (doc=195689316), train=1291 → test=1292 (doc=56893404)
- Old full nDCG@10 = 0.7909, old duplicate-filtered nDCG@10 = 0.7894

## Phase 3 — Triple Quality Knobs

### New flags in `scripts/train_bge_small_scifact.py`

| Flag | Default | Options |
|------|---------|---------|
| `--triple-source` | both | both, labeled-only, pseudo-only |
| `--max-pseudo-queries` | 5000 | any int |
| `--negatives-per-positive` | 3 | any int |
| `--disagreement-top-k` | 10 | any int |
| `--hard-negative-strategy` | disagreement | disagreement, bm25-top, dense-top, bm25+dense-top |

### New logic
- **Triple source ablation:** Can train with labeled-only, pseudo-only, or both
- **Hard negative strategies:** 4 strategies for collecting negatives (symmetric difference, BM25 top-k, dense top-k, union of both)
- **Gold doc exclusion:** All negatives are filtered to exclude gold documents
- **Triple dedup:** Exact (query, positive, negative) deduplication after generation
- **Parameterized disagreement range:** Configurable `--disagreement-top-k` (was hardcoded 10)

### New file
- `scripts/run_bge_small_phase3_sweep.py` — 9-config grid: baseline, labeled-only, pseudo-only, n=5 negs, top-20 disagreement, 4 hard-negative strategies, pq=10K

### Grid configs
1. `baseline_both_n3_dk10_hndisag` — reference
2. `labeled_only` — ablation: no pseudo triples
3. `pseudo_only` — ablation: no labeled triples
4. `n5` — 5 negatives per positive
5. `dk20` — top-20 disagreement range
6. `hnbm25` — BM25 top as hard negatives
7. `hndense` — dense top as hard negatives
8. `hnboth` — BM25 ∪ dense top as hard negatives
9. `pq10k` — 10K pseudo queries

## Phase 1 — Rebuild Clean Baseline (via `run_bge_small_clean_sweep`)
- Config `baseline_e5_lr2e5_wr01_bs32` = Phase 1 clean baseline
- Trains with `--exclude-train-query-ids 871,1291`, 5 epochs, lr=2e-5, warmup=0.1, batch=32

## Phase 2 — Sweep (via `run_bge_small_clean_sweep`)
- Remaining 5 configs = Phase 2 grid: e3/e7, lr 1e-5/3e-5, warmup 0.05
- All tuned on dev, test frozen

## Phase 4 — Curriculum 2-Stage

### New flags in `scripts/train_bge_small_scifact.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--curriculum` | off | Enable 2-stage training |
| `--curriculum-stage1-epochs` | 3 | Epochs on pseudo triples |
| `--curriculum-stage2-epochs` | 3 | Epochs on labeled triples |

### Logic
- Stage 1: train on pseudo triples only (broad, self-supervised)
- Stage 2: train on labeled triples only (clean, supervised)
- One-shot baseline unchanged

### New file
- `scripts/run_bge_small_phase4_sweep.py` — 4-config grid: one-shot baseline + 3 curriculum variants (s1e3+s2e3, s1e5+s2e3, s1e3+s2e5)

## Phase 5 — Tune RRF k/Weight

### Modified: `src/seg_retrieval/fusion.py`
- Added `weighted_rrf()` — RRF with per-retriever weights

### New file: `scripts/tune_rrf_scifact.py`
- Sweeps k ∈ {10,20,40,60,100} × bm25_weight ∈ {0.3,0.5,0.7,0.9}
- Dense weight fixed at 1.0 (relative)
- Selects best by dev nDCG@10
- `--final-test` runs best config on test exactly once
- `--dry-run` to verify grid (20 configs)

## Phase 6 — Cross-Dataset Robustness

### New file: `scripts/eval_cross_dataset.py`
- Evaluates fine-tuned checkpoint on SciFact + NFCorpus + FiQA + SciDocs
- Outputs per-dataset run CSV to `runs/<dataset>/test_dense_<model>.csv`
- Outputs summary JSON to `runs/cross_dataset_<model>.json`
- Handles both BEIR-format (`nfcorpus`, `scidocs`) and direct-format datasets

Usage:
```bash
python scripts/eval_cross_dataset.py --model-path runs/finetuned/bge-small-final
python scripts/eval_cross_dataset.py --model-path runs/finetuned/bge-small-final --datasets scifact nfcorpus
```

## Phase 7 — Finalize

### New file: `scripts/finalize_best_model.py`
Orchestrates the final protocol in 5 steps:
1. Leak audit with `--fail-on-leak`
2. Retrain best config with `--eval-split test` + fixed seed
3. Duplicate-filtered metric via audit `--run-csv`
4. Cross-dataset eval on all 4 BEIR datasets
5. Save `final.json` with all results

Supports:
- `--config` + `--best-name` to auto-load best from sweep summary
- Or override all parameters directly
- `--curriculum` for Phase 4 best configs
- `--dry-run` to verify all 5 steps

Usage:
```bash
# From sweep summary (recommended)
python scripts/finalize_best_model.py \
  --config runs/finetuned/bge-small-clean-sweep/summary.json \
  --best-name baseline_e5_lr2e5_wr01_bs32 \
  --seed 42

# Or manual override
python scripts/finalize_best_model.py \
  --output-dir runs/finetuned/bge-small-final \
  --epochs 5 --learning-rate 2e-5 --warmup-ratio 0.1 \
  --triple-source both --negatives-per-positive 3 \
  --curriculum --curriculum-stage1-epochs 5 --curriculum-stage2-epochs 3 \
  --seed 42
```

## Protocol Guarantees

- All sweeps use `--eval-split dev` — test is frozen
- Leak audit runs before every sweep
- Known source leaks (871, 1291) are excluded from training
- Every run saves `manifest.json` and `results.json`
- `finalize_best_model.py` runs test exactly once with fixed seed

## All Files in Scope

| File | Phase | Purpose |
|------|-------|---------|
| `scripts/audit_split_leakage.py` | 0 | 4 leak checks, `--fail-on-leak` |
| `scripts/train_bge_small_scifact.py` | 0-4 | Training with all flags |
| `scripts/run_bge_small_clean_sweep.py` | 1+2 | 6-config baseline + sweep |
| `scripts/run_bge_small_phase3_sweep.py` | 3 | 9-config triple quality |
| `scripts/run_bge_small_phase4_sweep.py` | 4 | 4-config curriculum |
| `scripts/tune_rrf_scifact.py` | 5 | 20-config RRF tuning |
| `scripts/eval_cross_dataset.py` | 6 | 4-dataset cross-eval |
| `scripts/finalize_best_model.py` | 7 | 5-step finalize protocol |
| `src/seg_retrieval/fusion.py` | 5 | Added `weighted_rrf()` |

## Full Workflow

```bash
# 1. Audit leaks
python scripts/audit_split_leakage.py --data-dir data/scifact

# 2. Sweep (all uses --eval-split dev)
python scripts/run_bge_small_clean_sweep.py       # Phase 1+2: 6 configs
python scripts/run_bge_small_phase3_sweep.py      # Phase 3: 9 configs
python scripts/run_bge_small_phase4_sweep.py      # Phase 4: 4 configs
python scripts/tune_rrf_scifact.py                # Phase 5: 20 configs

# 3. Choose best from summary CSVs

# 4. Finalize (test exactly once)
python scripts/finalize_best_model.py \
  --config <sweep-summary>.json \
  --best-name <best-config-name> \
  --seed 42

# Or for RRF final test:
python scripts/tune_rrf_scifact.py --final-test
```
