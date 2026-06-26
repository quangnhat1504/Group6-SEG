# SEG Phase 1-2 Progress Report

Date: 2026-05-26
Notebook: Researching SEG

## 1. Current Project Status

The SEG project is currently past the core Phase 1 and Phase 2 implementation milestones.

Completed:
- Phase 1 base retrieval pipeline on SciFact.
- BM25, Dense, and Hybrid RRF baselines.
- Dense model ablation with SciNCL, SPECTER, and MiniLM.
- Oracle route label generation from per-query retrieval performance.
- Phase 2 router baselines: Random, Majority, Oracle, Classical TF-IDF Logistic Regression.
- Small LLM router experiment using `Qwen/Qwen2.5-0.5B-Instruct` with QLoRA and label log-probability scoring.
- Held-out LLM score calibration and confidence-gated fallback.
- Phase 2 quality-vs-cost comparison table.
- Results have been written to Google Sheet `SEG_Result` for tracking.

Partially completed:
- Phase 3 uncertainty/selective reranking code exists, but the static Always-Rerank baseline and threshold ablation are still pending.

Next recommended task:
- Implement Always-Rerank baseline on Hybrid top-k using `cross-encoder/ms-marco-MiniLM-L-6-v2`, then run selective reranking threshold ablation.

## 2. Phase 1: Base Retrieval

Dataset:
- SciFact, BEIR-style format.
- Test split: 300 queries.
- Corpus: 5,183 documents.
- Retrieval depth: top-100.
- Main metrics: nDCG@10, Recall@10, Recall@100, MRR@10.

Implemented retrieval methods:
- BM25 lexical retrieval.
- Dense retrieval using SentenceTransformers.
- Hybrid retrieval using Reciprocal Rank Fusion (RRF).

### Phase 1 Main Results

| Method | Model | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---:|---:|---:|---:|
| BM25 | lexical | 0.6523 | 0.7757 | 0.8731 | 0.6184 |
| Dense | `malteos/scincl` | 0.5640 | 0.7233 | 0.9082 | 0.5224 |
| Hybrid RRF | BM25 + SciNCL | 0.6583 | 0.8146 | 0.9560 | 0.6157 |

Observations:
- BM25 is a strong baseline on SciFact test, with higher nDCG@10 and MRR@10 than Dense/SciNCL.
- Dense/SciNCL has better Recall@100 than BM25, suggesting that dense retrieval contributes useful candidate coverage.
- Hybrid RRF gives the best Phase 1 nDCG@10 and Recall@100 among the main retrieval baselines.
- The hybrid result supports using RRF as the default high-recall candidate generator for later reranking experiments.

### Dense Model Ablation

| Method | Dense Model | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---|---:|---:|---:|---:|
| Dense | `allenai/specter` | 0.3523 | 0.5004 | 0.7552 | 0.3133 |
| Hybrid RRF | BM25 + SPECTER | 0.3863 | 0.5562 | 0.7804 | 0.3421 |
| Dense | `sentence-transformers/all-MiniLM-L6-v2` | 0.6451 | 0.7833 | 0.9250 | 0.6047 |
| Hybrid RRF | BM25 + MiniLM | 0.4683 | 0.7102 | 0.9120 | 0.3987 |

Observations:
- SPECTER performed poorly in this setup.
- MiniLM dense retrieval was surprisingly competitive and close to BM25 on nDCG@10.
- However, Hybrid RRF with MiniLM underperformed the main BM25 + SciNCL Hybrid RRF. This suggests that dense-only quality does not automatically translate to better fusion.
- The current report should keep SciNCL as the official dense baseline because it was the planned scientific dense model and produced the strongest hybrid recall profile.

## 3. Oracle Route Labels

Oracle labels were created by choosing, for each query, the retrieval route with the best per-query nDCG@10 among:
- `bm25`
- `dense`
- `hybrid`

Test oracle distribution:
- BM25: 226 queries
- Dense: 49 queries
- Hybrid: 25 queries

Interpretation:
- The test split is strongly BM25-heavy.
- This distribution explains why the Majority Router, which always chooses BM25, is difficult to beat.
- It also reveals a train/test distribution shift, because the train oracle distribution was Dense-heavy:
  - BM25: 195
  - Dense: 477
  - Hybrid: 137

This distribution shift is one of the main reasons the Small LLM router struggled to generalize.

## 4. Phase 2: Query Routing

Goal:
- Select the best retrieval route per query before returning results.
- Compare simple lower bounds, an oracle upper bound, a classical ML router, and a Small LLM router.

Implemented routers:
- Random Router.
- Majority Router.
- Oracle Router.
- Classical TF-IDF + Logistic Regression Router.
- Small LLM QLoRA Router using `Qwen/Qwen2.5-0.5B-Instruct`.

Small LLM input format:
- The model receives only the query text wrapped in an instruction prompt.
- It does not receive retrieved documents, BM25 scores, dense scores, or metadata.
- Output labels are restricted to: `bm25`, `dense`, `hybrid`.
- The latest version uses label log-probability scoring rather than free-text generation.

### Phase 2 Router Results

| Router | Accuracy | Macro-F1 | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Random Router | 0.3267 | 0.2617 | 0.6290 | 0.7778 | 0.9132 | 0.5875 |
| Majority Router | 0.7533 | 0.2864 | 0.6523 | 0.7757 | 0.8731 | 0.6184 |
| Oracle Router | 1.0000 | 1.0000 | 0.7617 | 0.8711 | 0.9127 | 0.7337 |
| Classical TF-IDF LogReg Router | 0.9933 | 0.9857 | 0.7617 | 0.8711 | 0.9160 | 0.7337 |
| Small LLM QLoRA Router LogProb | 0.4300 | 0.3076 | 0.6372 | 0.7788 | 0.9046 | 0.5980 |
| Small LLM QLoRA Calibrated Held-Out | 0.3133 | 0.2097 | 0.6674 | 0.8161 | 0.8947 | 0.6259 |

Important caveat:
- The Classical TF-IDF LogReg Router was trained and evaluated on the same test split, so it should be treated as an in-split sanity baseline, not as a final generalization result.
- The calibrated held-out LLM row tunes on 150 test queries and evaluates on the other 150. It avoids tuning and evaluating on the same queries, but it is still a constrained diagnostic because no separate Colab-generated dev prediction file is available yet.

Observations:
- Oracle routing shows large headroom over static retrieval, with nDCG@10 = 0.7617.
- Majority Router is strong because test oracle labels are BM25-heavy.
- Small LLM QLoRA improved after switching to label log-probability scoring, but it still trails the Majority Router in nDCG@10.
- Confidence-gated fallback improves the LLM route's retrieval quality on held-out queries, reaching nDCG@10 = 0.6674 on 150 evaluation queries.
- The LLM router tends to overpredict Dense relative to the BM25-heavy test oracle distribution.
- Routing alone is currently not the strongest contribution unless calibration, confidence gating, or retrieval-aware features are added.

### Phase 2 Quality-vs-Cost Table

The generated CSV `runs/scifact/test_phase2_quality_cost.csv` estimates retrieval cost with simple units:
- BM25 = 1
- Dense = 3
- Hybrid = 4

Key rows:

| Method | Matched Queries | Cost Units / Query | Route Distribution | nDCG@10 | Recall@10 | Recall@100 |
|---|---:|---:|---|---:|---:|---:|
| BM25 | 300 | 1.00 | bm25 | 0.6523 | 0.7757 | 0.8731 |
| Hybrid RRF | 300 | 4.00 | hybrid | 0.6583 | 0.8146 | 0.9560 |
| Small LLM QLoRA LogProb | 300 | 2.31 | bm25=113, dense=169, hybrid=18 | 0.6372 | 0.7788 | 0.9046 |
| Small LLM QLoRA Calibrated Held-Out | 150 | 3.16 | bm25=42, hybrid=108 | 0.6674 | 0.8161 | 0.8947 |

Interpretation:
- Static BM25 remains the strongest low-cost baseline.
- Static Hybrid RRF gives better recall at higher cost.
- Raw LLM routing reduces estimated cost relative to Hybrid but loses nDCG@10.
- Calibrated fallback can exceed Hybrid nDCG@10 on held-out queries, but the comparison uses only 150 evaluation queries and should be presented as follow-up evidence rather than the main final claim.

## 5. Interpretation and Research Direction

The strongest current result is not the Small LLM router by itself. The evidence points toward this project being stronger as an uncertainty-aware multi-stage retrieval system:

1. Use Hybrid RRF as a strong candidate generator.
2. Use routing and retrieval disagreement signals to estimate uncertainty.
3. Apply cross-encoder reranking only when uncertainty is high.
4. Compare against Always-Rerank to show the effectiveness-cost trade-off.

This direction is more defensible than relying only on Small LLM routing, because:
- Static Hybrid RRF is already strong.
- Oracle routing shows headroom.
- Small LLM routing is sensitive to class imbalance and distribution shift.
- Selective reranking can produce a clearer cost-quality contribution.

## 6. Remaining Work

Phase 2 improvement tasks:
- Phase 2 follow-up implementation is complete for the current available prediction files.
- Optional: generate a separate dev prediction CSV from Colab to replace the held-out test-query calibration diagnostic with a cleaner train/dev/test LLM calibration setup.

Phase 3 tasks:
- Implement Always-Rerank static baseline on Hybrid top-k.
- Run threshold ablation for selective reranking.
- Report rerank coverage, latency/query, nDCG@10, Recall@10, and MRR@10.

Report deliverables:
- Table 1: BM25 vs Dense vs Hybrid retrieval.
- Table 2: Random vs Majority vs Oracle vs Classical Router vs Small LLM Router.
- Table 3: BM25 vs Dense/SciNCL vs Hybrid RRF vs Always-Rerank vs Proposed Selective Reranking.
- Related Work section covering BM25+RM3, SPLADE/DeepImpact, ColBERT, Listwise LLM reranking, and GraphRouter/RouterDC.
- Final paper-style report with motivation, method, experiments, ablation, limitations.

## 7. Summary

Phase 1 is complete. The project has a working SciFact retrieval pipeline with BM25, Dense/SciNCL, Hybrid RRF, and dense ablation results.

Phase 2 core and follow-up work are complete for the available local artifacts. Router baselines, Small LLM routing, score calibration, confidence-gated fallback, retrieval diagnostics, and quality-vs-cost comparison have been implemented and evaluated. The results show that routing has potential, but the Small LLM router is still limited by class imbalance and distribution shift.

The next high-priority step is Phase 3: build Always-Rerank and run selective reranking ablations. This will turn the project from a routing-only experiment into a stronger uncertainty-aware retrieval system with a measurable quality-cost trade-off.
