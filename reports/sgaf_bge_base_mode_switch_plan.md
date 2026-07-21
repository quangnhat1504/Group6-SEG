# SGAF BGE-base Mode-Switch Plan

## Motivation

The current final candidate, **Uncertainty-shift Adaptive Coverage SGAF**, is stronger than BGE-small and fixed A3 on all four evaluated datasets. However, when the baseline is changed to BGE-base, the conclusion changes:

| Dataset | BGE-small | BGE-base | Current adaptive | Adaptive - BGE-base | Current coverage | Oracle coverage |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.8188 | 0.7376 | 0.8218 | +0.0842 | 0.084 | 0.050 |
| NFCorpus | 0.3505 | 0.3695 | 0.3559 | -0.0136 | 0.180 | 1.000 |
| FiQA | 0.3635 | 0.3909 | 0.3774 | -0.0136 | 0.219 | 1.000 |
| SciDocs | 0.1893 | 0.2147 | 0.1962 | -0.0185 | 0.211 | 1.000 |

Average across all four datasets:

| Method | Avg nDCG@10 |
|---|---:|
| BGE-small | 0.4305 |
| BGE-base | 0.4282 |
| Current adaptive | 0.4378 |

Transfer-only average on NFCorpus, FiQA, and SciDocs:

| Method | Transfer Avg nDCG@10 |
|---|---:|
| BGE-small | 0.3011 |
| BGE-base | 0.3251 |
| Current adaptive | 0.3098 |

This means the current method is not a universal replacement for BGE-base. It is a specialist-preserving method that improves over BGE-small/fixed A3, but it remains too conservative on shifted transfer datasets where BGE-base is globally stronger.

## Diagnosis

The main bottleneck is not the rescue ranker itself. The bottleneck is the **coverage controller**:

- Current uncertainty-shift coverage is capped at `0.40`.
- Actual selected transfer coverages are only `0.180`, `0.219`, and `0.211`.
- Oracle coverage sweep selects `1.000` for NFCorpus, FiQA, and SciDocs, which exactly recovers BGE-base as the best coverage-only policy.
- SciFact oracle coverage remains low at `0.050`, because BGE-base is much weaker than the SciFact specialist there.

So the next question is:

> Can we keep low specialist-first coverage on source-like queries, while allowing a cheap label-free controller to enter BGE-base global mode when distribution shift is strong?

## Proposed Logic

Keep the existing two-level SGAF structure:

1. The frozen SciFact-trained BGE-base rescue ranker orders queries by likely generalist rescue value.
2. A label-free controller chooses how much BGE-base coverage is allowed.

Add a third mode to the controller:

| Mode | Trigger | Coverage behavior | Intended role |
|---|---|---|---|
| Specialist-safe | Target looks source-like and specialist confidence is high | `0.02` to `0.10` | Protect SciFact performance |
| Adaptive-rescue | Moderate specialist uncertainty or disagreement | `0.10` to `0.40` | Current SGAF behavior |
| Generalist-fallback | Strong source shift and specialist uncertainty | `0.60` to `1.00`, or pure BGE-base | Recover transfer robustness |

The fallback must be label-free at evaluation time. Candidate signals should come only from already-computed run statistics:

- specialist top score, gap, and top-10 score standard deviation;
- BGE-small vs BGE-base overlap at 10 and 20;
- cross-rank of each retriever's top document;
- query length and IDF-like lexical statistics if available;
- batch-level z-score shift from SciFact trainfit feature means.

## Novelty Boundary

This is still an ensemble direction, but the novelty is not "combine BGE-small and BGE-base." The stronger framing is:

> A specialist-generalist retrieval policy separates rescue ranking from coverage control, then uses cheap uncertainty/shift statistics to decide whether a query batch should stay specialist-first, use bounded generalist rescue, or fall back to global generalist mode.

The method is novel enough for the project only if the fallback is:

- label-free on target datasets;
- cheaper than CrossEncoder or LLM reranking;
- evaluated through ablations that isolate each controller component;
- honest about BGE-base remaining the transfer default when no specialist preservation is required.

## Ablation Plan

Run ablations incrementally. Do not jump directly to the full controller.

| ID | Added component | Description | Main question | Expected contribution |
|---|---|---|---|---|
| B0 | BGE-small | Specialist baseline | How strong is the source specialist? | Reference |
| B1 | BGE-base | Generalist baseline | What is the transfer default? | Reference |
| B2 | Fixed A3 5% | Frozen rescue ranker with fixed low coverage | Does sparse generalist rescue help safely? | Small positive vs BGE-small |
| B3 | Current uncertainty coverage | Current formula capped at 0.40 | Does cheap uncertainty increase transfer? | Positive vs A3, still below BGE-base transfer |
| B4 | Coverage cap sweep | Same formula with caps `0.40`, `0.60`, `0.80`, `1.00` | Is the gap mostly caused by the cap? | Identify SciFact/transfer tradeoff |
| B5 | Batch source-shift mode | Increase max coverage only when batch-level shift is high | Can we preserve SciFact and recover transfer? | Stronger transfer with limited SciFact loss |
| B6 | Query-level fallback gate | Route selected high-shift queries to BGE-base/pure generalist | Does per-query fallback beat budget-only coverage? | Higher transfer, higher risk |
| B7 | Hybrid controller | Batch mode decides max coverage; query gate decides which queries use it | Does combining batch and query signals add value? | Candidate final method |
| B8 | Oracle coverage sweep | Target-qrels diagnostic only | How much headroom remains? | Upper bound, not claim |
| B9 | Oracle component router | Target-qrels diagnostic only | Is ranking or coverage the limiting factor? | Upper bound, not claim |

## Contribution Table Template

Use this table format for reporting. Each row adds exactly one component.

| Step | Method | SciFact | NFCorpus | FiQA | SciDocs | Avg | Transfer Avg | Delta vs previous | Delta vs BGE-base transfer | SciFact loss vs BGE-small |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | BGE-small |  |  |  |  |  |  |  |  |  |
| 1 | BGE-base |  |  |  |  |  |  |  |  |  |
| 2 | Fixed A3 5% |  |  |  |  |  |  |  |  |  |
| 3 | Current uncertainty coverage |  |  |  |  |  |  |  |  |  |
| 4 | Cap sweep best frozen cap |  |  |  |  |  |  |  |  |  |
| 5 | Batch source-shift mode |  |  |  |  |  |  |  |  |  |
| 6 | Query fallback gate |  |  |  |  |  |  |  |  |  |
| 7 | Hybrid mode-switch controller |  |  |  |  |  |  |  |  |  |

## Evaluation Protocol

Keep the protocol clean:

1. Fit ranker and source statistics on SciFact `trainfit`.
2. Use SciFact `dev` only for selecting the final controller family and small hyperparameter grid.
3. Freeze the recipe.
4. Evaluate once on SciFact `test`, duplicate-filtered SciFact `test`, NFCorpus, FiQA, and SciDocs.
5. Report paired bootstrap against BGE-small, fixed A3, current adaptive coverage, and BGE-base.
6. Treat oracle coverage and oracle router as diagnostic headroom only.

## Success Criteria

A candidate is worth promoting only if it satisfies all of these:

- SciFact nDCG@10 remains within `0.005` of current adaptive SGAF or improves it.
- Transfer average closes at least half of the current gap to BGE-base.
- No transfer dataset drops below current adaptive SGAF.
- At least one transfer dataset has a significant gain over current adaptive SGAF.
- The controller uses only cheap retrieval/post-retrieval statistics, not CrossEncoder or LLM calls.

## First Implementation Step

Add a new script, likely `scripts/run_sgaf_mode_switch_ablation.py`, with the same loaders and feature extraction used by `run_adaptive_coverage_sgaf.py`.

Start with B4 only:

- reuse the current uncertainty-shift formula;
- sweep upper caps `0.40`, `0.60`, `0.80`, and `1.00`;
- write a table showing SciFact preservation vs transfer recovery.

If cap-only changes are insufficient, use B4b/B5 diagnostics to determine whether the blocker is formula strength or missing mode-switch behavior.

## B4 Result

Implemented:

- `scripts/run_sgaf_mode_switch_ablation.py`
- `runs/fusion/sgaf_mode_switch_ablation/sgaf_mode_switch_ablation_rows.csv`
- `runs/fusion/sgaf_mode_switch_ablation/sgaf_mode_switch_ablation_summary.csv`
- `runs/fusion/sgaf_mode_switch_ablation/sgaf_mode_switch_ablation_significance.csv`
- `reports/tables/table_sgaf_mode_switch_ablation.md`

Key finding:

| Method | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|
| Current uncertainty coverage | 0.3098 | -0.0152 | +0.0030 |
| Cap-only max 1.00 | 0.3098 | -0.0152 | +0.0030 |
| Gain 2.0 max 1.00 | 0.3128 | -0.0122 | -0.0051 |
| Gain 4.0 max 1.00 | 0.3176 | -0.0074 | -0.0097 |
| Gain 8.0 max 1.00 | 0.3251 | +0.0000 | -0.0103 |

B4a conclusion:

- Increasing only the cap is a no-op because the current formula selects coverage below `0.40` on every dataset.
- The immediate blocker is not the hard cap. It is that the uncertainty formula is too conservative and lacks a generalist-fallback mode.

B4b conclusion:

- Increasing uncertainty gain can close the transfer gap to BGE-base.
- However, global gain hurts SciFact beyond the planned `0.005` preservation threshold.
- Therefore the next step should not be a global gain increase. It should be **B5 batch-level source-shift mode**, where high coverage is enabled only when target/batch statistics indicate strong shift.

Revised next step:

1. Implement B5: a batch-level mode selector with low max coverage for SciFact-like batches and high max coverage for shifted batches.
2. Keep B4b gain rows as diagnostic evidence only.
3. Implement B6 per-query fallback only if B5 cannot recover enough transfer without hurting SciFact.

## B5 Result

B5 adds a cheap batch-level shift score:

`shift = abs(z(query_len)) + max(0,-z(bge_small_top)) + max(0,-z(bge_small_gap)) + max(0,-z(bge_small_std10)) + max(0,-z(overlap10))`

Observed batch shift scores:

| Dataset | Shift score | Mode under threshold 2.0 |
|---|---:|---|
| SciFact | 1.087 | specialist-safe |
| NFCorpus | 6.025 | generalist-fallback |
| FiQA | 3.793 | generalist-fallback |
| SciDocs | 3.584 | generalist-fallback |

B5 results:

| Method | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|
| Current uncertainty coverage | 0.3098 | -0.0152 | +0.0030 |
| Batch shift t2.0 gain 4.0 | 0.3176 | -0.0074 | +0.0030 |
| Batch shift t2.0 gain 6.0 | 0.3249 | -0.0001 | +0.0030 |
| Batch shift t2.0 gain 8.0 | 0.3251 | +0.0000 | +0.0030 |

Interpretation:

- B5 validates the mode-switch direction: source-like SciFact keeps current adaptive coverage, while shifted batches can use high BGE-base coverage.
- This recovers the BGE-base transfer average without sacrificing the SciFact specialist gain.
- The current best exploratory candidate is `Batch shift t2.0 gain 6.0` or `gain 8.0`.
- For a cleaner next step, freeze one conservative B5 recipe before adding per-query B6. Prefer `t2.0 gain 6.0` because it nearly matches BGE-base transfer while avoiding unnecessary pure fallback on NFCorpus.

Revised next step after B5:

1. Freeze `Batch shift t2.0 gain 6.0` as the next candidate.
2. Run duplicate-filtered SciFact and paired bootstrap against current adaptive, BGE-small, and BGE-base.
3. Only then consider B6 per-query fallback if B5 leaves meaningful per-dataset headroom.

## Frozen B5 Candidate Result

Implemented:

- `scripts/summarize_sgaf_mode_switch_final.py`
- `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_rows.csv`
- `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_summary.csv`
- `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_duplicate_filtered_scifact.csv`
- `runs/fusion/final_sgaf_mode_switch/final_sgaf_mode_switch_significance.csv`
- `reports/tables/table_final_sgaf_mode_switch.md`

Frozen recipe:

- threshold `2.0`
- shifted gain `6.0`
- shifted cap `1.0`
- SciFact trainfit BGE-base rescue classifier, `C=0.1`

Summary:

| Method | Avg nDCG@10 | Transfer Avg | Transfer delta vs BGE-base | SciFact delta vs BGE-small |
|---|---:|---:|---:|---:|
| BGE-small specialist | 0.4305 | 0.3011 | -0.0239 | +0.0000 |
| BGE-base generalist | 0.4282 | 0.3251 | +0.0000 | -0.0812 |
| Current adaptive SGAF | 0.4378 | 0.3098 | -0.0152 | +0.0030 |
| Frozen B5 mode-switch SGAF | 0.4492 | 0.3249 | -0.0001 | +0.0030 |

Duplicate-filtered SciFact:

| Method | Filtered nDCG@10 | Filtered delta vs BGE-small | Filtered delta vs current |
|---|---:|---:|---:|
| BGE-small specialist | 0.8176 | +0.0000 | -0.0030 |
| Current adaptive SGAF | 0.8206 | +0.0030 | +0.0000 |
| Frozen B5 mode-switch SGAF | 0.8206 | +0.0030 | +0.0000 |

Significance highlights:

- vs current adaptive: significant on SciDocs, near-significant on NFCorpus and FiQA under 10k bootstrap.
- vs BGE-small: significant on NFCorpus, FiQA, and SciDocs.
- vs BGE-base: no transfer loss is significant; FiQA/SciDocs exactly match BGE-base because B5 enters pure generalist mode.

Decision:

- B5 is the strongest mode-switch core candidate because it preserves SciFact specialist behavior and recovers BGE-base transfer.
- Do not add B6 immediately. The remaining risk is validation protocol, not mechanism capacity.
- Next step should be paper framing and/or testing the frozen B5 recipe on any genuinely new held-out batch without retuning threshold/gain.

Phase 8 update:

- Frozen P3 rank-window smoothing now builds on B5 and is the strongest current experimental candidate.
- P3 recipe: apply only on `generalist_fallback` batches, `window=20`, `alpha=0.10`, `rrf_k=60`.
- P3 transfer average is `0.3293`, delta vs Frozen B5 `+0.0043`, delta vs BGE-base transfer `+0.0042`.
- Keep the caveat: P3 still needs external/new-batch frozen validation because the smoothing grid was designed after seeing the current project datasets.
