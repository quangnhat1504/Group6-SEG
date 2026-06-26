# SEG Phase 3 Progress Update - 2026-06-20

## Phase 3: Selective Reranking - Current Status

### Completed
1. **Always-Rerank baseline** with CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`
2. **GPU acceleration** for cross-encoder inference (RTX 5070 Ti 16GB)
3. **Updated `run_selective_rerank.py`** with `--rerank-all` mode

### Always-Rerank Results (Hybrid top-20, GPU):
- nDCG@10: 0.6939
- Recall@10: 0.8286
- Recall@100: 0.9560
- MRR@10: 0.6604
- Rerank coverage: 1.0 (100% queries)
- Latency: 27ms/query (vs 405ms CPU)

### Performance Comparison
- **CPU**: 405ms/query
- **GPU**: 27ms/query (≈14x speedup)
- **Cross-encoder**: MiniLM-L-6-v2 (lightweight for student hardware)

### Technical Implementation
1. **GPU detection**: Auto-detects CUDA availability, falls back to CPU
2. **Always-Rerank mode**: Uses Hybrid run as candidate set, reranks all queries
3. **Configuration override**: Command-line args for model and top_k

### Next Steps (Pending)
1. **Threshold ablation**: Vary router_confidence, score_gap, disagreement thresholds
2. **Pareto efficiency plot**: Effectiveness (nDCG@10) vs Cost (latency + rerank coverage)
3. **Selective reranking experiments**: Find optimal threshold balancing quality and cost
4. **Table 3**: BM25 vs Dense vs Hybrid vs Always-Rerank vs Selective Reranking

### Project State
- Phase 0, 1, 2 fully completed
- Phase 3.1 (Always-Rerank) completed
- Phase 3.2 (Uncertainty triggers) already implemented
- Pending: 3.3 (threshold ablation), 3.4 (Pareto plot), 3.5 (report update)

### Hardware Notes
- Installed PyTorch with CUDA 12.8 for RTX 5070 Ti
- GPU working effectively for cross-encoder inference
- Student hardware constraint met (fast inference on local GPU)

### Code Updates
- `scripts/run_selective_rerank.py`: Added `--rerank-all`, `--rerank-top-k`, `--cross-encoder`
- `src/seg_retrieval/rerank.py`: Added device detection and GPU support
- Tasks.md: Updated to reflect Phase 3 progress

### Timeline
- 2026-06-20: Always-Rerank baseline completed on GPU
- Target: Complete threshold ablation and final report within next session