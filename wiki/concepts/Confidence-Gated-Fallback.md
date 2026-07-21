---
title: "Confidence-Gated Fallback"
type: concept
tags: [calibration, routing, fallback, llm]
sources: [phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
A calibration technique for the Small LLM router: when the LLM's label margin (top-1 minus top-2 log-probability) is below a threshold, the system falls back to Hybrid RRF instead of trusting the uncertain route prediction.

## Configuration
- Temperature: 1.0
- Margin threshold: 0.3
- Fallback label: Hybrid
- Class biases: BM25=+0.75, Dense=+0.25, Hybrid=0.0

## Performance
- nDCG@10: 0.6674 on 150 held-out queries
- Cost: 3.16 units/query (many queries fall back to Hybrid)
- Trade-off: improves quality but increases cost
- Mean Regret@10: 0.0708 (lowest among compared methods)

## Related To
- [[Query Routing]] — the application
- [[QLoRA]] — model fine-tuning
- [[Qwen2.5-0.5B-Instruct]] — LLM backbone
- [[Distribution Shift]] — reason calibration is needed
