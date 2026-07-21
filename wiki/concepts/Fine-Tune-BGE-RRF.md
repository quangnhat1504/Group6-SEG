# Fine-tune BGE with RRF Disagreement (vstash recipe)

- **Goal:** Tạo model embedding fine-tune của riêng mình (không clone), dùng vstash self-supervised recipe
- **Recipe:** RRF disagreement signal giữa BM25 và dense → symmetric difference của top-10 → negative pool; RRF top-3 → positives; train MNRL
- **65K triples** generated from SciFact train (45K pseudo + 2.7K labeled)
- **Proven reference:** Stffens/bge-small-rrf-v3 (33M) achieves 0.7707 on SciFact — beats Oracle Router (0.7617) and BGE-base (0.7376)

## Result: SUCCESS ✔
- **BGE-small (33M) fine-tuned on SciFact → 0.7909 nDCG@10**
- **Beats:** BGE-base (109M, 0.7376) by +0.0533, vstash rrf-v3 (0.7707) by +0.0202, Oracle Router (0.7617) by +0.0292
- **Full metrics:** MAP@10=0.7479, MRR@10=0.7564, R@10=0.9113, R@100=0.9783
- **Training:** 47,241 triples (45K pseudo + 2.7K labeled), 5 epochs, MNRL, ~16 min on RTX 5070 Ti
- **Script:** `scripts/train_bge_small_scifact.py`
- **Model:** `runs/finetuned/bge-small-scifact-rrf/`

## Previous failures
- **BGE-base (109M):** Failed 3× with catastrophic forgetting (−0.0135 to −0.0399). MNRL too aggressive for large pretrained model. Need lower LR + more freeze + different loss (see [[Catastrophic Forgetting]]).

## Next: Use as a Specialist
The fine-tuned BGE-small model is now best understood as a **SciFact specialist**, not a universal replacement for BGE-base or vstash. The next main direction is [[Specialist-Generalist Adaptive Fusion]]: combine this specialist with BM25 and generalist dense retrievers per query.

Multi-corpus fine-tuning remains a useful ablation, but not the primary novelty claim.

## Key insight
- Fine-tuning model nhỏ (33M) hiệu quả và ổn định. Scaling lên model lớn (109M) cần recipe khác — không đơn giản là tăng model size.
- RRF disagreement signal là self-supervised, không cần labeled data.
- **Updated novelty:** the model provides a strong specialist component. The research novelty should come from leakage-aware specialist/generalist fusion and transfer analysis, not merely from owning a fine-tuned checkpoint.

## Leak Audit (Phase 0, 2026-07-11)

- **2 known source leaks** detected in SciFact train: query 871 (text matches test 870), query 1291 (text matches test 1292)
- Both excluded from training via `--exclude-train-query-ids 871,1291`
- `scripts/audit_split_leakage.py` checks 4 leak types, `--fail-on-leak` for CI

## Triple Quality Knobs (Phase 3, 2026-07-11)

| Flag | Description |
|------|-------------|
| `--triple-source` | both / labeled-only / pseudo-only |
| `--max-pseudo-queries` | pseudo query cap (default 5000) |
| `--negatives-per-positive` | negs per positive (default 3) |
| `--disagreement-top-k` | disagreement range (default 10) |
| `--hard-negative-strategy` | disagreement / bm25-top / dense-top / bm25+dense-top |

Dedup added: exact (query, positive, negative) triples removed after generation.

## Date updated: 2026-07-11