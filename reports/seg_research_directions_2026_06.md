# SEG — Hướng Nghiên Cứu Mở Rộng (Deep Research, 2026-06-21)

Mục tiêu: tìm hướng đi mới, sáng tạo, có tiềm năng "publishable" cho hệ thống truy hồi tài liệu khoa học đa tầng, nhận-biết-bất-định, tiết-kiệm-chi-phí trên SciFact — nhưng vẫn khả thi trong ràng buộc GPU 16GB / 3 tháng / small LLM.

Phương pháp: fan-out web search (8 truy vấn) trên literature 2023-2026, đối chiếu nhiều nguồn. Đây là tổng hợp + xếp hạng, KHÔNG phải bê nguyên gợi ý của NotebookLM.

Luận điểm cốt lõi (thesis) cần được củng cố:
> "Dùng routing rẻ + tín hiệu bất định để quyết định KHI NÀO các tầng đắt tiền (rerank, LLM) đáng giá."

---

## Bảng xếp hạng nhanh (novelty × feasibility × coherence với thesis)

| # | Hướng | Novelty | Feasibility | Coherence | Effort |
|---|---|---|---|---|---|
| 1 | Conformal / risk-controlled selective reranking | Cao | Cao | **Hoàn hảo** | Thấp-TB |
| 2 | QPP làm tín hiệu gating thống nhất (route + rerank) | TB-Cao | Cao | Cao | Thấp-TB |
| 3 | Downstream-utility reranking (semantic entropy của small LLM verify claim) | Cao | TB | Cao | TB |
| 4 | Small listwise LLM reranker (FIRST-style) làm tầng đắt thứ 3 | TB-Cao | TB | Cao | TB-Cao |
| 5 | (Phụ trợ) GPL domain-adaptation cho SciNCL + abstention cho NEI | Thấp-TB | Cao | TB | TB |

Loại khỏi shortlist (lý do bên dưới): Knowledge-triple/OpenIE fusion, Pre-route reasoning.

---

## Hướng 1 — Conformal / Risk-Controlled Selective Reranking ⭐ (KHUYẾN NGHỊ CHÍNH)

(a) **Pitch:** Thay vì chọn ngưỡng uncertainty bằng grid-search cảm tính, dùng **conformal risk control / Learn-Then-Test (LTT)** để chọn ngưỡng có **bảo đảm thống kê**: "rerank ít nhất bao nhiêu % query để giữ risk (vd mất nDCG@10, hoặc tỉ lệ bỏ sót oracle) ≤ α với xác suất 1−δ", hoặc ngược lại "tối đa chất lượng dưới ngân sách chi phí GPU cho trước".

(b) **Vì sao mới:** Selective reranking hiện tại của nhóm chỉ là threshold heuristic. Đưa conformal vào biến nó thành hệ có **chứng minh toán học** — đúng tinh thần các paper rất mới: *Conformal Ranked Retrieval* (arXiv 2404.17769), *Two-stage / Selective Conformal Risk Control* (arXiv 2512.12844), *Streamlining Conformal IR via Score Refinement* trên BEIR (arXiv 2410.02914). Áp dụng conformal cho **quyết định KHI NÀO rerank** (chứ không phải cho prediction set) là góc còn ít người làm → novelty thật.

(c) **Cách làm (khớp SciFact + 16GB):** Thuần CPU, không train thêm. (1) Tách 300 test query thành calibration/test sạch (hoặc dùng train split). (2) Định nghĩa nonconformity score từ tín hiệu sẵn có (router margin, BM25 score-gap, RRF agreement). (3) Dùng LTT/CRC để tìm ngưỡng λ thỏa risk bound. (4) Vẽ **risk-coverage curve** và đối chiếu Always-Rerank (0.6939) vs no-rerank Hybrid (0.6583).

(d) **Thêm vào report:** Một định lý/thủ tục + **risk-coverage figure** + bảng "rerank coverage cần thiết để đạt X% của Always-Rerank gain". Đây là phần "ăn điểm" về nền tảng lý thuyết.

(e) **Effort:** Thấp-TB. **Risk:** conformal cần exchangeability — train/test distribution shift của nhóm vi phạm giả định này → nhưng chính điều đó là một *finding* hay (báo cáo coverage gap dưới shift).

(f) **Củng cố thesis:** Đây chính là phiên bản có-bảo-đảm của thesis: chứng minh được "rerang chọn lọc đạt gần Always-Rerank với chi phí thấp hơn, có guarantee".

---

## Hướng 2 — QPP làm tín hiệu gating thống nhất

(a) **Pitch:** Nâng cấp các "retrieval-aware diagnostics" rời rạc (score-gap, overlap) thành một **Query Performance Prediction (QPP)** có tên gọi, dùng QPP để quyết định CẢ route lẫn có-rerank-hay-không.

(b) **Vì sao mới:** Hiện nhóm dùng các tín hiệu ad-hoc. Framework QPP cho neural retrieval đang là hướng nóng: Faggioli et al. (ICTIR 2023, "periodic table" of NIR-QPP), ADG-QPP cho dense (Springer ML 2024/2025), QPP-GenRE (LLM-based, 2024), workshop QPP++ 2025. Đặc biệt có nghiên cứu QPP cho **routing/variant-selection trong RAG** (TREC-RAG 2024). Đóng góp mới: QPP như **bộ điều khiển chi phí đa tầng**.

(c) **Cách làm:** Tính các predictor rẻ từ run sẵn có — NQC, σ_max, score-gap chuẩn hóa, query-feedback, ensemble disagreement BM25↔Dense. Tương quan với per-query nDCG@10 (đã có oracle labels!). Dùng predictor mạnh nhất làm tín hiệu cho Hướng 1.

(d) **Thêm vào report:** Bảng correlation (Kendall-τ / Pearson) giữa từng QPP predictor và per-query nDCG; scatter QPP vs gain-from-reranking.

(e) **Effort:** Thấp-TB. **Risk:** literature cảnh báo QPP cho dense có tương quan yếu và phụ thuộc collection (Faggioli 2023) → cần đặt kỳ vọng thực tế; nhưng "QPP nào dự báo gain-from-rerank tốt nhất" tự nó là câu hỏi nghiên cứu.

(f) **Củng cố thesis:** Cho thesis một "bộ não" định lượng: tín hiệu rẻ nào thực sự dự báo được khi nào tầng đắt đáng giá.

---

## Hướng 3 — Downstream-Utility Reranking (semantic entropy của small LLM)

(a) **Pitch:** Rerank tài liệu KHÔNG theo relevance bề mặt, mà theo **mức độ nó giúp một small LLM xác minh claim** (SciFact vốn là claim verification). Tài liệu làm LLM trả lời nhất quán hơn (giảm semantic entropy) → cộng điểm.

(b) **Vì sao mới:** Đây là hướng "hot" nhất 2024-2026: **Semantic Entropy** (Farquhar et al., *Nature* 2024), **LLM-Confidence Reranker / MSCP** (arXiv 2602.13571), **InfoGain-RAG** (Document Information Gain, arXiv 2509.12765), **LURE-RAG** utility-driven reranking. Áp vào **truy hồi khoa học trên SciFact** và **gate bằng uncertainty** (chỉ chạy khi cross-encoder không chắc) là đóng góp mới và ăn khớp downstream task của SciFact (SUPPORTS/REFUTES/NEI).

(c) **Cách làm:** Dùng Qwen2.5-0.5B/1.5B (đã có trong stack). Với mỗi candidate top-k: prompt "claim + abstract → SUPPORT/REFUTE/NEI", sample N lần, tính semantic entropy / MSCP; rerank theo độ nhất quán. Vì đắt → chỉ bật cho query mà Hướng 1/2 báo "không chắc".

(d) **Thêm vào report:** Bảng so Always-Rerank (cross-encoder) vs Utility-Rerank vs Selective-Utility-Rerank; figure entropy vs đúng/sai. Một phân tích định tính ví dụ.

(e) **Effort:** TB. **Risk:** small LLM confidence có thể nhiễu trên biomedical; chi phí sampling cao (đúng tinh thần "đắt nên phải gate"). Cần kiểm soát số sample.

(f) **Củng cố thesis:** Đẩy thesis sang tầng đắt nhất & ý nghĩa nhất: nối retrieval với utility thực sự ở cuối pipeline, và chỉ trả giá khi cần.

---

## Hướng 4 — Small Listwise LLM Reranker (FIRST-style) làm tầng đắt thứ 3

(a) **Pitch:** Mở rộng cascade thành 3 tầng: BM25/Hybrid → cross-encoder MiniLM → **listwise LLM reranker** chỉ cho query khó. Dùng **FIRST** (single-token decoding) để rẻ hơn listwise thường.

(b) **Vì sao mới:** FIRST (EMNLP 2024, arXiv 2406.15657) đạt nDCG@10 ≈ **75.3 trên SciFact** (so với 72.3 của generation thường) và nhanh hơn ~50%; còn có reproduction (arXiv 2411.05508). So sánh cross-encoder vs LLM reranker trên SPLADE (arXiv 2403.10407) khung "contenders trong phổ hiệu quả-chi phí". Đóng góp mới: đặt LLM reranker làm **tầng được kích hoạt chọn lọc** chứ không phải luôn bật.

(c) **Cách làm:** Dùng checkpoint FIRST/RankZephyr quantize 4-bit (vừa 16GB nhưng chậm) HOẶC một setwise reranker ≤3B. Chỉ chạy listwise trên ~top-20 của số query mà selective-gate đánh dấu.

(d) **Thêm vào report:** Hàng "Selective LLM-rerank" trong Table 3 + điểm trên Pareto effectiveness-vs-latency (3 điểm: no-rerank, cross-encoder, LLM-rerank).

(e) **Effort:** TB-Cao. **Risk:** 7B-4bit có thể quá chậm trên 16GB → cân nhắc model nhỏ hơn; là hướng nặng nhất.

(f) **Củng cố thesis:** Minh họa rõ nhất phổ chi phí 3 tầng và giá trị của việc gate tầng đắt nhất.

---

## Hướng 5 (phụ trợ) — GPL domain-adaptation cho SciNCL + Abstention cho NEI

(a) **Pitch:** Sửa điểm yếu thật của hệ (Dense/SciNCL kém: nDCG 0.5640, lại gây train/test shift) bằng **GPL** (Generative Pseudo Labeling, NAACL 2022; R-GPL re-mining hard negatives, arXiv 2501.14434). Kèm cơ chế **abstention** ánh xạ tự nhiên vào nhãn **NEI/NoInfo** của SciFact.

(b) **Vì sao đáng làm:** GPL chỉ cần corpus không nhãn, tăng tới ~9 nDCG trên domain shift, chạy được 16GB. Abstention khi bất định cao là câu chuyện cost-efficiency + chống ảo giác đẹp, và NEI là nhãn có sẵn trong SciFact → không gượng ép.

(c) **Cách làm:** Sinh synthetic query từ abstract → mine hard negative → cross-encoder pseudo-label → fine-tune SciNCL. Abstention: nếu sau mọi tầng uncertainty vẫn cao → trả "no sufficient evidence".

(d/e/f):** Bảng "SciNCL gốc vs GPL-adapted" + ablation; effort TB, risk: có thể không cải thiện nhiều vì Hybrid đã mạnh → để làm ablation/secondary, không phải đóng góp chính.

---

## Loại khỏi shortlist

- **Knowledge-triple / OpenIE fusion (gợi ý #3 của NotebookLM):** OpenIE trên biomedical rất nhiễu, thêm pipeline nặng, gắn yếu với thesis cost/uncertainty. Các paper KG-claim mới (GraphCheck 2502.16514, GraphFC 2503.07282) chủ yếu cho *verification* dùng LLM lớn, không phải *retrieval* nhẹ → lệch scope. Effort cao, payoff bấp bênh.
- **Pre-route structured reasoning (gợi ý #4):** Chủ yếu lợi về explainability; bằng chứng cải thiện accuracy cho LLM 0.5B yếu. Giữ như một ablation nhỏ nếu dư thời gian, không phải hướng chính.

---

## KHUYẾN NGHỊ CUỐI

**Primary = Hướng 1 (Conformal risk-controlled selective reranking)** — đây là "xương sống": nó hoàn tất đúng phần Phase 3 còn dang dở (threshold ablation + Pareto), nhưng nâng lên tầm có-bảo-đảm-thống-kê → vừa khả thi cao vừa novelty thật vừa khớp 100% thesis.

**Secondary = Hướng 3 (Downstream-utility / semantic-entropy reranking)** — là phần "sáng tạo & publishable" nhất: cung cấp một tín-hiệu/tầng-rerank mới gắn retrieval với chính nhiệm vụ verify claim của SciFact; vì đắt nên được **gate bởi Hướng 1** → câu chuyện thống nhất.

**Chất keo = Hướng 2 (QPP)** — dùng để chọn nonconformity score cho Hướng 1 và quyết định khi nào gọi Hướng 3.

> Narrative thống nhất cho báo cáo cuối: *"Multi-stage retrieval nhận-biết-bất-định, trong đó tín hiệu QPP rẻ + conformal guarantee quyết định KHI NÀO gọi tầng đắt; và tầng đắt nhất là downstream-utility reranker chấm tài liệu theo mức nó giúp small LLM xác minh claim."* — novelty × feasibility × coherence đều cao.

Lộ trình đề xuất: hoàn tất threshold ablation đang dở → bọc bằng conformal (H1) → thêm QPP predictors (H2) → nếu còn thời gian, làm utility-rerank gated (H3). H4/H5 là mở rộng/ablation.

---

## Nguồn chính (đã đối chiếu)

- Conformal/risk control IR: arXiv 2404.17769 (Conformal Ranked Retrieval), 2512.12844 (Selective Conformal Risk Control), 2410.02914 (Streamlining Conformal IR / BEIR).
- QPP: Faggioli et al. ICTIR 2023 (NIR-QPP framework); ADG-QPP (Springer ML 2024/25); QPP-GenRE 2024; QPP++ 2025; QPP-for-RAG-routing (TREC-RAG 2024).
- Downstream-utility/semantic entropy: Farquhar et al. *Nature* 2024 (Semantic Entropy); LLM-Confidence Reranker/MSCP (arXiv 2602.13571); InfoGain-RAG (2509.12765); LURE-RAG.
- Listwise LLM rerank: FIRST (EMNLP 2024, 2406.15657; reproduction 2411.05508); cross-encoder vs LLM on SPLADE (2403.10407); Shallow Cross-Encoders (2403.20222).
- Adaptive/cost-aware retrieval: Self-RAG (Asai 2024), CRAG (Yan 2024), Adaptive-RAG (Jeong 2024), SeaKR (2406.19215), TARG (2511.09803).
- Query expansion gated: HyDE, Query2doc; "Not All Queries Need Rewriting" (2603.13301); CSQE (2402.18031); AQE (2507.11042).
- Domain adaptation: GPL (NAACL 2022, 2112.07577); R-GPL re-mining (2501.14434).
- KG claim verification (đã loại): GraphCheck (2502.16514), GraphFC (2503.07282).
