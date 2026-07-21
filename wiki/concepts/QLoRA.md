---
title: "QLoRA"
type: concept
tags: [fine-tuning, quantization, lora, llm]
sources: [phase1-phase2-report, seg-research-paper]
last_updated: 2026-06-29
---

## Summary
QLoRA (Quantized Low-Rank Adaptation) enables memory-efficient fine-tuning of LLMs with 4-bit quantization and LoRA adapters. Used in SEG to fine-tune [[Qwen2.5-0.5B-Instruct]] for query routing on Google Colab with limited VRAM.

## Usage in SEG
- Fine-tunes 0.5B parameter model as query router
- Trained on train split oracle labels (809 queries)
- Initial run: free-text generation, failed to predict BM25
- Second run: label log-probability scoring, improved accuracy
- Third follow-up: undersampling for class balance
- Calibration: class biases, temperature scaling, margin threshold

## Related To
- [[Qwen2.5-0.5B-Instruct]] — LLM backbone
- [[Query Routing]] — application
- [[Confidence-Gated Fallback]] — post-training calibration
