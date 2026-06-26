"""Property-based tests for RRF fusion validity.

Feature: seg-experimental-validation, Property 4: RRF Fusion Validity

Validates: Requirements 3.2 (implicitly via pipeline correctness)
"""
from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

# Add src/ to path so we can import seg_retrieval
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from seg_retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from seg_retrieval.types import Run  # noqa: E402


@composite
def valid_run(draw: st.DrawFn) -> Run:
    """Generate a valid Run dict with 1-20 queries, each having 1-50 hits."""
    num_queries = draw(st.integers(min_value=1, max_value=20))
    run: Run = {}
    for i in range(num_queries):
        query_id = f"q{i}"
        num_hits = draw(st.integers(min_value=1, max_value=50))
        # Generate unique doc_ids for this query
        doc_ids = [f"d{i}_{j}" for j in range(num_hits)]
        # Generate random scores (RRF ignores input scores, uses rank only)
        scores = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=num_hits,
                max_size=num_hits,
            )
        )
        run[query_id] = list(zip(doc_ids, scores))
    return run


@settings(max_examples=100)
@given(
    run_a=valid_run(),
    run_b=valid_run(),
    k=st.integers(min_value=1, max_value=200),
    top_k=st.integers(min_value=1, max_value=100),
)
def test_rrf_fusion_validity(run_a: Run, run_b: Run, k: int, top_k: int) -> None:
    """Property 4: RRF Fusion Validity.

    For any two valid non-empty runs and any positive integer k,
    reciprocal_rank_fusion([run_a, run_b], k=k, top_k=top_k) produces a run where:
      (a) for each query, the result list is sorted by score in non-increasing order
      (b) every doc_id in the output for a query appears in at least one input run for that query
      (c) the output query set equals the union of query sets from both inputs

    **Validates: Requirements 3.2**
    """
    fused = reciprocal_rank_fusion([run_a, run_b], k=k, top_k=top_k)

    # Property 4a: For each query, results are sorted by score in non-increasing order
    for query_id, hits in fused.items():
        scores = [score for _, score in hits]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Query {query_id}: scores not sorted at index {i}: "
                f"{scores[i]} < {scores[i + 1]}"
            )

    # Property 4b: Every doc_id in output appears in at least one input run for that query
    for query_id, hits in fused.items():
        input_doc_ids: set[str] = set()
        for doc_id, _ in run_a.get(query_id, []):
            input_doc_ids.add(doc_id)
        for doc_id, _ in run_b.get(query_id, []):
            input_doc_ids.add(doc_id)
        for doc_id, _ in hits:
            assert doc_id in input_doc_ids, (
                f"Query {query_id}: doc_id '{doc_id}' not in any input run"
            )

    # Property 4c: Output query set equals the union of input query sets
    expected_queries = set(run_a.keys()) | set(run_b.keys())
    actual_queries = set(fused.keys())
    assert actual_queries == expected_queries, (
        f"Output queries {actual_queries} != expected union {expected_queries}"
    )
