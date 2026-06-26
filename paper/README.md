# SEG Paper — Uncertainty-Aware Cost-Efficient Multi-Stage Retrieval (SciFact)

Thư mục này chứa bản report dạng **academic paper** (tiếng Việt, giữ nguyên thuật ngữ tiếng Anh),
viết bằng LaTeX, sẵn sàng upload lên **Prism / Overleaf**.

## Cấu trúc thư mục

```
paper/
├── main.tex                 # File chính: preamble tiếng Việt + TikZ workflow + \input các section
├── README.md                # File này
├── sections/                # Nội dung từng chương (mỗi file một \input)
│   ├── abstract.tex
│   ├── introduction.tex     # Giới thiệu + Câu hỏi nghiên cứu (RQ1–RQ5)
│   ├── related_work.tex
│   ├── method.tex           # Method + sơ đồ workflow TikZ (Hình 1)
│   ├── experiments.tex      # Thiết lập thí nghiệm, metrics, cost proxy
│   ├── results.tex          # Toàn bộ bảng + 3 hình kết quả
│   ├── discussion.tex       # Ablation, diagnostics + Discussion
│   ├── limitations.tex
│   ├── conclusion.tex       # Next work + Conclusion
│   └── appendix.tex         # Reproducibility + NotebookLM
├── figures/                 # Ảnh PNG cho các hình kết quả
│   ├── phase3_pareto.png
│   ├── phase3_conformal_risk_coverage.png
│   └── qpp_correlations.png
└── references/
    └── references.bib       # 19 tài liệu tham khảo (BibTeX)
```

> **Lưu ý về "folder chứa reference paper":** `references/references.bib` chứa metadata + link arXiv
> của mọi citation. Nếu muốn lưu kèm bản PDF gốc của từng paper, hãy tạo `references/pdfs/` và bỏ
> file PDF vào đó (không bắt buộc để biên dịch).

## Cách biên dịch

Paper dùng **pdfLaTeX** + **bibtex** với hỗ trợ tiếng Việt qua `babel` (vietnamese) và `fontenc` T5.

### Trên Prism / Overleaf
1. Nén toàn bộ thư mục `paper/` thành `.zip` rồi upload (hoặc kéo-thả).
2. Đặt **Main document** = `main.tex`.
3. Compiler = **pdfLaTeX** (mặc định). Bấm Recompile 1–2 lần để cập nhật mục lục và trích dẫn.

### Local (terminal)
```bash
cd paper
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

## Ghi chú nội dung
- **Hình 1** (workflow) được vẽ bằng **TikZ** ngay trong `method.tex` — không cần file ảnh ngoài.
- Mọi bảng dùng `booktabs` để trông chuyên nghiệp.
- Số liệu trong các bảng/hình lấy trực tiếp từ `reports/` của repo (đồng bộ tại thời điểm 22/06/2026).
- Nếu Prism dùng XeLaTeX và báo lỗi font T5, đổi sang pdfLaTeX, hoặc thay 3 dòng font trong `main.tex`
  bằng `fontspec` + `polyglossia` (mục `\setmainlanguage{vietnamese}`).
