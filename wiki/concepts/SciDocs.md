# SciDocs

- **Source:** BEIR benchmark (citation prediction subset)
- **Scale:** 25,657 documents, 1,000 test queries
- **Domain:** Scientific citation prediction — paper titles as queries, full-text papers as documents
- **Query style:** Short paper titles → find relevant papers to cite
- **Challenge:** Title queries không khớp từ vựng với full-text papers. Khác biệt nhất trong 4 BEIR datasets.
- **BM25 baseline:** nDCG@10 = 0.1495 (rất yếu)
- **SciNCL:** nDCG@10 = 0.1951 (+30.5% so với BM25)
- **BGE-base:** nDCG@10 = 0.2147 (+0.0196, p=0.008)
- **Adaptive RRF:** nDCG@10 = 0.2097 (-0.0050, n.s.)
- **CE on ARF:** nDCG@10 = 0.1910 (-0.0187, p<0.001, có ý nghĩa — CE gây hại)
- **Key insight:** Margin BGE vs SciNCL nhỏ nhất trong 4 dataset (+0.0196 vs +0.1737 trên SciFact), cho thấy citation prediction ít hưởng lợi từ modern embeddings hơn.
- **Date added:** 2026-07-01