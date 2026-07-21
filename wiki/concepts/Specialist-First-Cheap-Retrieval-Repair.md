---
title: "Specialist-First Cheap Retrieval Repair"
type: concept
tags: [retrieval, post-processing, cheap-repair, specialist]
last_updated: 2026-07-20
---

# Specialist-First Cheap Retrieval Repair

Specialist-First Cheap Retrieval Repair is a cheaper alternative to full Specialist-Generalist Adaptive Fusion.

The default retriever is the fine-tuned BGE-small specialist. The system only applies post-retrieval repair when the specialist appears weak. The first repair source is BM25 because it is cheap and can recover entity-heavy or lexical-specific failures without running an extra dense model, Cross-Encoder, or LLM.

## Motivation

BGE-small-final is the strongest SciFact specialist, but analysis shows a subset of queries where BM25 can still rescue specialist failures. On SciFact test, BM25 strictly beats the specialist on `32/300` queries, and oracle selection over BM25 + BGE-small-final reaches `0.8624`, or `+0.0436` above BGE-small-final.

The challenge is not whether headroom exists. The challenge is detecting rescueable failures without labels.

## Pipeline Shape

1. Retrieve with BGE-small-final.
2. Compute cheap confidence signals:
   - specialist top score;
   - specialist top1-top2 gap;
   - specialist top-10 score spread;
   - BM25 top score/gap;
   - BM25-specialist overlap.
3. If specialist confidence is high, keep the specialist ranking.
4. If confidence is low, apply a bounded cheap repair:
   - rerank inside specialist candidates with BM25 features;
   - or inject only a small number of BM25 candidates.
5. Optional cheap learned gate:
   - train a small logistic gate from specialist/BM25 agreement features;
   - apply rank-local promotion only when the gate predicts BM25 rescue.

## Current Evidence

See [[cheap-post-retrieval-repair-ablation]].

Current result: useful diagnostic headroom exists. Hand-written rules are not robust enough. A learned cheap gate is useful diagnostically, but clean validation does not meet the success threshold.

Dev multi-seed summary:

- BGE-small-final mean nDCG@10: `0.9123`
- BM25 rerank inside specialist top-N: `0.9124`
- Conditional BM25 injection: `0.9119`
- Always BM25 injection: `0.9047`
- Bounded BM25 promotion: `0.9121`
- Rank-local BM25 promotion: `0.9116`
- Learned cheap gate: `0.9123` as no-op, because dev calibration has no BM25 rescue-positive labels

Test diagnostic multi-seed summary:

- BGE-small-final mean nDCG@10: `0.8200`
- BM25 rerank inside specialist top-N: `0.8183`
- Conditional BM25 injection: `0.8167`
- Conditional lexical BM25 rescue: `0.8184`
- Bounded BM25 promotion: `0.8193`
- Rank-local BM25 promotion: `0.8205`
- Learned cheap gate + rank-local promotion: `0.8248`

Clean trainfit->dev validation:

- BGE-small-final dev nDCG@10: `0.9052`
- S8 rank-local promotion: `0.9052`
- S9 learned cheap gate: `0.9057`
- S9 delta: `+0.0005`, below the `+0.005` success criterion
- Trainfit BM25 rescue-positive queries: `2`
- Dev BM25 rescue-positive queries: `0`

## Leader Interpretation

Do not claim cheap repair improves the specialist yet.

The current defensible claim is narrower:

> A BM25-rescue failure region exists, simple score-threshold and RRF-injection rules do not reliably exploit it, and a cheap learned gate can recover part of the headroom on diagnostic test-derived splits.

The clean validation shows the practical weakness: there are too few BM25 rescue-positive labels in trainfit and none on dev, so the cheap BM25 repair path cannot provide a robust main contribution under the current split.

## Next Research Step

The next useful ablation is no longer another BM25-only repair rule. Move to Specialist-Generalist Adaptive Fusion:

- run static specialist/generalist ensemble validation on dev;
- inspect whether BGE-base/vstash rescue regions exist where BM25 does not;
- only then train query-adaptive fusion.

Keep S9 as a secondary low-cost baseline and diagnostic section.
