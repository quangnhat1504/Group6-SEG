---
title: "SPECTER"
type: concept
tags: [dense-model, embedding, scientific, citation-informed]
sources: [phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
SPECTER (Scientific Paper Embeddings using Citation-informed Transformers) is a dense retriever model trained on citation graphs for scientific documents. It was tested as a dense retriever ablation in SEG but significantly underperformed.

## Performance on SciFact
- Dense only: nDCG@10=0.3523, Recall@10=0.5004
- Hybrid with BM25: nDCG@10=0.3863
- Substantially below BM25 (0.6523) and SciNCL (0.5640)

## Conclusion
Despite being designed for scientific document representations, SPECTER performs poorly for SciFact retrieval. The project therefore keeps [[SciNCL]] as the primary scientific dense model.

## Related To
- [[Dense Retrieval]] — retrieval approach
- [[SciNCL]] — preferred alternative
- [[MiniLM]] — general-domain alternative
- [[BM25 Retrieval]] — lexical baseline that outperforms SPECTER
