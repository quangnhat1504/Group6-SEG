# vstash / Stffens/bge-small-rrf-v3

- **Source:** https://github.com/stffns/vstash (MIT license)
- **Model:** [Stffens/bge-small-rrf-v3](https://huggingface.co/Stffens/bge-small-rrf-v3) — BGE-small (33M, 384d) fine-tuned with self-supervised disagreement signal from RRF pipeline
- **Paper:** arXiv:2604.15484 (April 2026) — "vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents"
- **Key result (paper):** Beats ColBERTv2 on 5/5 BEIR datasets. vstash SciFact NDCG@10 = 0.7263 (BGE-small + full pipeline with MMR)
- **Our experiment (2026-06-30):** vstash rrf-v3 standalone on SciFact test = **0.7707** — exceeds Oracle Router (0.7617) and BGE-base (0.7376). Verified with full metrics: MAP=0.7227, R@10=0.9110, R@100=0.9900, MRR=0.7316.
- **Cross-dataset (2026-06-30):** NFCorpus = 0.3667 (~BGE-base 0.3695). FiQA = 0.4825 (>> BGE-base 0.3909, +0.0916). SciDocs = 0.2150 (≈ BGE-base 0.2147).
- **Our own SciFact specialist:** BGE-small final reaches **0.8188 nDCG@10** on SciFact, but transfers worse than vstash on FiQA/NFCorpus/SciDocs. This makes vstash a useful generalist/reference model rather than a defeated baseline.
- **Current hypothesis:** The best next step is not simply scaling the same fine-tune recipe to BGE-base; it is [[Specialist-Generalist Adaptive Fusion]] over BM25, BGE-base, the SciFact specialist, and optionally vstash.
- **Recipe reference:** MNRL on RRF disagreement triples, with leak-audited training exclusions for SciFact source duplicates.
- **Date updated:** 2026-07-11