# SGAF Phase 8C Duplicate/Canonical Collapse Ablation

This is a cheap post-retrieval ablation. It removes repeated `doc_id` or repeated normalized document text within each query ranking, keeping the highest-ranked occurrence.

## Summary

| Method | Datasets | Removed hits | Affected queries | Mean delta nDCG@10 | Max abs delta | Min delta | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BGE-small specialist | 4 | 436 | 239 | +0.0001 | +0.0003 | +0.0000 | diagnostic_only |
| BGE-base generalist | 4 | 545 | 258 | -0.0001 | +0.0006 | -0.0006 | diagnostic_only |
| Current adaptive SGAF | 4 | 423 | 216 | +0.0003 | +0.0010 | +0.0000 | diagnostic_only |
| Frozen B5 mode-switch SGAF | 4 | 523 | 245 | -0.0001 | +0.0005 | -0.0005 | diagnostic_only |

## Dataset Detail

| Dataset | Method | Removed hits | Affected queries | Baseline nDCG@10 | Collapsed nDCG@10 | Delta nDCG@10 | Delta Recall@100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | BGE-small specialist | 0 | 0 | 0.8188 | 0.8188 | +0.0000 | +0.0000 |
| scifact | BGE-base generalist | 0 | 0 | 0.7376 | 0.7376 | +0.0000 | +0.0000 |
| scifact | Current adaptive SGAF | 0 | 0 | 0.8218 | 0.8218 | +0.0000 | +0.0000 |
| scifact | Frozen B5 mode-switch SGAF | 0 | 0 | 0.8218 | 0.8218 | +0.0000 | +0.0000 |
| nfcorpus | BGE-small specialist | 410 | 213 | 0.3505 | 0.3508 | +0.0003 | -0.0020 |
| nfcorpus | BGE-base generalist | 544 | 257 | 0.3695 | 0.3690 | -0.0006 | -0.0034 |
| nfcorpus | Current adaptive SGAF | 421 | 214 | 0.3559 | 0.3570 | +0.0010 | -0.0026 |
| nfcorpus | Frozen B5 mode-switch SGAF | 522 | 244 | 0.3692 | 0.3687 | -0.0005 | -0.0034 |
| fiqa | BGE-small specialist | 24 | 24 | 0.3635 | 0.3635 | +0.0000 | +0.0000 |
| fiqa | BGE-base generalist | 0 | 0 | 0.3909 | 0.3909 | +0.0000 | +0.0000 |
| fiqa | Current adaptive SGAF | 0 | 0 | 0.3774 | 0.3774 | +0.0000 | +0.0000 |
| fiqa | Frozen B5 mode-switch SGAF | 0 | 0 | 0.3909 | 0.3909 | +0.0000 | +0.0000 |
| scidocs | BGE-small specialist | 2 | 2 | 0.1893 | 0.1893 | +0.0000 | +0.0000 |
| scidocs | BGE-base generalist | 1 | 1 | 0.2147 | 0.2147 | +0.0000 | +0.0000 |
| scidocs | Current adaptive SGAF | 2 | 2 | 0.1962 | 0.1962 | +0.0000 | +0.0000 |
| scidocs | Frozen B5 mode-switch SGAF | 1 | 1 | 0.2147 | 0.2147 | +0.0000 | +0.0000 |

## Interpretation

- If all rows are no-op, duplicate/canonical collapse is not a BEIR benchmark improvement source.
- The same logic can still be useful for the production web app if retrieval returns multiple chunks from one canonical source.
- Treat `diagnostic_only` rows as evidence-cleaning or UI-diversity candidates, not as Frozen B5 performance contributions.
