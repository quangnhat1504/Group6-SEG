# FiQA

- **Source:** BEIR benchmark (financial question answering subset)
- **Scale:** 57,638 documents, 648 test queries
- **Domain:** Financial QA — colloquial user questions vs financial text
- **Query style:** Ngôn ngữ đời thường, không khớp với văn bản tài chính
- **Challenge:** Dataset khó nhất cho BM25 vì query dùng informal language
- **BM25 baseline:** nDCG@10 = 0.2167
- **SciNCL:** nDCG@10 = 0.0800 (rất yếu — domain shift nặng)
- **BGE-base:** nDCG@10 = 0.3909 (+0.3110, p<0.001 — cải thiện lớn nhất)
- **Adaptive RRF:** nDCG@10 = 0.3827 (-0.0082, n.s. — BM25 quá yếu, tự động giảm weight về 0)
- **vstash rrf-v3:** nDCG@10 = 0.4825 (+0.0916 so với BGE-base)
- **Key insight:** Semantic matching quan trọng nhất. Adaptive RRF không giúp vì BM25 quá yếu.
- **Date added:** 2026-07-01