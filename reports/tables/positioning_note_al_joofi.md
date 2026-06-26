# Positioning Note: SEG vs. Al-Joofi et al. (MDPI Applied Sciences, 2025)

> **Reference:** Al-Joofi et al., "An Empirical Investigation of Multi-Stage
> Scientific Paper Retrieval with SciFact," *Applied Sciences* (MDPI), 2025.

This note documents the relationship between the SEG (Selective Expertise Gating)
framework and the findings of Al-Joofi et al. Both studies investigate multi-stage
retrieval pipelines on the SciFact dataset, but they differ substantially in scope,
evaluation protocol, and research contributions. Direct comparison of absolute metric
values between the two studies is inappropriate and should not be attempted.

---

## 1. Shared Findings

The following empirical conclusions are corroborated by both studies, providing
independent confirmation across different experimental configurations:

### 1.1 RRF k Dilution Effect

Both studies observe that **lower RRF k values strengthen the hybrid retrieval base**.
As k decreases from the commonly used k=60 toward smaller values (e.g., k=5), the
fused ranking improves because lower k increases the influence of highly-ranked
documents relative to those ranked further down. This "k dilution" effect is
consistent across both evaluation protocols.

### 1.2 Cross-Encoder Reranking as Primary Performance Driver

Both studies confirm that **cross-encoder reranking is the single largest source of
effectiveness gains** in multi-stage scientific retrieval pipelines. Regardless of the
base retrieval configuration, adding a neural reranker yields the most substantial
nDCG improvement. This holds across different reranker models and rerank depths.

### 1.3 Hybrid Fusion Outperforms Individual Retrievers

Both studies demonstrate that **hybrid fusion of BM25 and dense retrieval consistently
outperforms either retriever in isolation**. This finding validates the complementary
nature of lexical and semantic matching for scientific document retrieval.

---

## 2. Protocol Differences Preventing Direct Comparison

The following methodological differences between the two studies make direct comparison
of absolute metric values (e.g., nDCG@10 numbers) invalid:

| Aspect | SEG | Al-Joofi et al. |
|--------|-----|-----------------|
| **Test queries** | 300 queries (full SciFact test set) | 100 queries (SciFact subset) |
| **Dense retriever** | SciNCL (`malteos/scincl`) | SPECTER / SciBERT variants |
| **Corpus preprocessing** | Minimal (title + abstract concatenation) | Custom preprocessing pipeline |
| **Rerank depth** | Top-20 candidates | Top-100 candidates |
| **Evaluation split** | 150 calibration / 150 evaluation (seed=13) | Full test set evaluation |

**Critical note:** SEG evaluates on 300 queries while Al-Joofi evaluates on 100
queries. This difference alone is sufficient to invalidate direct metric comparison,
as nDCG@10 averages are sensitive to query set composition and size. The two studies
use different subsets of SciFact's test queries, different dense retrieval models with
different embedding spaces, and different reranking depths that affect how much of the
candidate list the cross-encoder can improve.

---

## 3. SEG's Novel Contributions Beyond Al-Joofi et al.

While Al-Joofi et al. provide a thorough empirical investigation of multi-stage
pipeline configurations, SEG introduces several novel components that are entirely
absent from their work:

### 3.1 Uncertainty-Aware Selective Gating

SEG introduces a **QPP-based decision mechanism** that determines, on a per-query
basis, whether cross-encoder reranking is likely to improve results. Queries where
the base retrieval is already confident (low uncertainty) skip the expensive reranking
stage entirely. This selective approach is unique to SEG and has no counterpart in
Al-Joofi et al.

### 3.2 QPP Signals for Predicting Reranking Utility

SEG identifies and validates **Query Performance Prediction (QPP) features** —
specifically `hybrid_max` — as effective predictors of when reranking will provide
gains. The feature selection is validated on a held-out train split to avoid data
leakage, and correlations are reported for both train and test partitions.

### 3.3 Conformal Risk Control Guarantees

SEG applies **Conformal Risk Control (CRC)** to provide formal statistical guarantees
on the expected nDCG shortfall when selectively skipping reranking. At alpha=0.02,
the framework guarantees that the expected loss from skipping reranking does not
exceed 2% of nDCG@10. This provides a principled, distribution-free threshold
selection mechanism with coverage guarantees — a capability absent from Al-Joofi
et al.'s pipeline.

### 3.4 Downstream-Utility Reranking via LLM Citation Worthiness

SEG explores **LLM-based utility reranking** where a language model assesses
citation-worthiness of retrieved documents for a given query. This downstream-utility
signal provides a complementary reranking approach that goes beyond relevance matching
to assess practical utility for scientific workflows.

### 3.5 Efficiency Analysis: Latency vs. Effectiveness Trade-offs

SEG provides a comprehensive **latency analysis** measuring the wall-clock cost of
each pipeline stage and demonstrating the efficiency gains from selective reranking.
By skipping reranking for high-confidence queries, SEG achieves meaningful latency
reductions while maintaining effectiveness within the CRC guarantee bounds. This
cost-effectiveness perspective is not explored in Al-Joofi et al.

---

## Summary

SEG and Al-Joofi et al. share foundational empirical conclusions about multi-stage
retrieval (RRF k effects, reranking dominance, hybrid superiority), providing mutual
corroboration. However, the studies differ fundamentally in scope: Al-Joofi et al.
focus on pipeline configuration and hyperparameter analysis, while SEG introduces
novel uncertainty-aware selective mechanisms with formal statistical guarantees. The
protocol differences (300 vs. 100 queries, different models, different depths) preclude
any valid direct numerical comparison between the two studies.
