---
title: "Paper — LaTeX Build Instructions"
type: source
tags: [latex, paper, build]
date: 2026-06-22
source_file: raw/paper-readme.md
---

## Summary
Build instructions for the SEG academic paper written in Vietnamese LaTeX (Tiếng Việt + English technical terms). Organized as modular `\input` sections with TikZ workflow diagram, 3 result figures (Pareto, conformal risk/coverage, QPP correlations), 19 BibTeX references, and booktabs-formatted tables.

## Key Claims
- Compile with pdfLaTeX + bibtex on Overleaf/Prism or locally
- Paper structured as: Abstract, Introduction (RQ1-RQ5), Related Work, Method, Experiments, Results, Discussion, Limitations, Conclusion, Appendix
- Vietnamese with babel (vietnamese) and fontenc T5
- All figures and tables sourced directly from `reports/` artifacts
- Bidirectional sync: if paper tables change, corresponding report markdown must stay in sync

## Connections
- [[SciFact]] — dataset
- [[BM25 Retrieval]] — method
- [[Dense Retrieval]] — method
- [[Reciprocal Rank Fusion]] — method
- [[Selective Reranking]] — method
- [[Conformal Risk Control]] — method
- [[Query Performance Prediction]] — method
