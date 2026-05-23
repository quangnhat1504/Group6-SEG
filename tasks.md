# SEG Research Task List

## Phase 0 - Project Setup
- [x] Create memory bank `.agent_scratchpad.md`.
- [x] Create Python package scaffold.
- [x] Add reproducible config for SciFact experiments.
- [x] Add README with environment and run commands.

## Phase 1 - Base Retrieval / Assignment 1
- [x] Load SciFact corpus, queries, and qrels through BEIR format.
- [x] Build BM25 lexical retrieval baseline.
- [x] Build dense retrieval baseline with SentenceTransformers + FAISS/NumPy fallback.
- [x] Run required dense baseline with `malteos/scincl`.
- [ ] Optionally run `allenai/specter` or `sentence-transformers/all-MiniLM-L6-v2` for dense ablation.
- [x] Implement RRF hybrid fusion.
- [x] Evaluate BM25, dense, and hybrid with nDCG@10, Recall@10/100, MRR@10.
- [x] Export per-query run files and per-query oracle route labels.

## Phase 2 - Query Routing / Assignment 2
- [x] Add Random Router lower-bound.
- [x] Add Majority Router lower-bound.
- [x] Add Oracle Router upper-bound.
- [x] Train classical TF-IDF + LinearSVC/LogReg router from oracle labels.
- [x] Evaluate router Accuracy and Macro-F1.
- [x] Add Small LLM QLoRA training notebook/script after baseline stabilizes.
- [x] Integrate router decision into retrieval pipeline.
- [x] Improve Small LLM router class balance/prompt after first QLoRA run underpredicted BM25.
- [ ] Replace free-text LLM generation with label logit scoring for bm25/dense/hybrid.

## Phase 3 - Uncertainty + Selective Reranking / Assignment 3
- [x] Implement uncertainty signals: router confidence, score gap, retriever disagreement.
- [x] Add cross-encoder reranker for top-k candidates.
- [x] Trigger reranking only when uncertainty exceeds threshold.
- [ ] Run threshold ablation and plot effectiveness vs cost.
- [ ] Report rerank coverage, latency/query, nDCG@10, Recall@10, MRR@10.

## Report Deliverables
- [ ] Table 1: BM25 vs Dense vs Hybrid retrieval.
- [ ] Table 2: Random vs Majority vs Oracle vs Classical Router vs Small LLM Router.
- [ ] Table 3: BM25 vs Dense/SciNCL vs Hybrid RRF vs Always-Rerank vs Proposed Selective Reranking.
- [ ] Related Work only: BM25+RM3, SPLADE/DeepImpact, ColBERT, Listwise LLM reranking, GraphRouter/RouterDC.
- [ ] Final paper-style report: motivation, method, experiments, ablation, limitations.
