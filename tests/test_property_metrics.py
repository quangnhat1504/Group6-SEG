"""Property-based tests for evaluation metrics completeness.

Feature: seg-experimental-validation, Property 6: Evaluation Metrics Completeness

Validates: Requirements 5.2
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to path so we can import seg_retrieval
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from seg_retrieval.metrics import evaluate_run


@composite
def run_and_qrels(draw):
    """Generate valid Run and Qrels with overlap.

    - Run: 1-30 queries, each with 1-50 hits as (doc_id, score) tuples with unique doc_ids
    - Qrels: each query has 1-5 relevant documents (relevance grade 1 or 2)
    - Ensures some qrels queries appear in the run (overlap)
    """
    n_queries = draw(st.integers(min_value=1, max_value=30))

    query_ids = [f"q{i}" for i in range(n_queries)]

    # Generate Run: each query has 1-50 hits with unique doc_ids
    run = {}
    for qid in query_ids:
        n_hits = draw(st.integers(min_value=1, max_value=50))
        doc_ids = draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=8,
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                ),
                min_size=n_hits,
                max_size=n_hits,
                unique=True,
            )
        )
        scores = draw(
            st.lists(
                st.floats(
                    min_value=0.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_hits,
                max_size=n_hits,
            )
        )
        # Sort by score descending
        hits = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
        run[qid] = hits

    # Generate Qrels: each query has 1-5 relevant documents with grade 1 or 2
    qrels = {}
    for qid in query_ids:
        n_rel = draw(st.integers(min_value=1, max_value=5))
        # Some relevant docs may overlap with the run's doc_ids
        rel_doc_ids = draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=8,
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                ),
                min_size=n_rel,
                max_size=n_rel,
                unique=True,
            )
        )
        grades = draw(
            st.lists(
                st.integers(min_value=1, max_value=2),
                min_size=n_rel,
                max_size=n_rel,
            )
        )
        qrels[qid] = dict(zip(rel_doc_ids, grades))

    return run, qrels


EXPECTED_KEYS = {"ndcg@10", "recall@10", "recall@100", "mrr@10"}


@settings(max_examples=100)
@given(data=run_and_qrels())
def test_evaluate_run_metrics_completeness(data):
    """Property 6: Evaluation Metrics Completeness.

    For any valid run (at least one query with at least one hit) and valid qrels
    (at least one query with at least one relevant document), evaluate_run returns
    a dict with exactly keys ndcg@10, recall@10, recall@100, mrr@10, each a float
    in [0.0, 1.0].

    **Validates: Requirements 5.2**
    """
    run, qrels = data

    result = evaluate_run(run, qrels)

    # Exactly 4 keys
    assert set(result.keys()) == EXPECTED_KEYS, (
        f"Expected keys {EXPECTED_KEYS}, got {set(result.keys())}"
    )

    # All values are floats in [0.0, 1.0]
    for key, value in result.items():
        assert isinstance(value, (int, float)), (
            f"Value for '{key}' is not numeric: {type(value)}"
        )
        assert 0.0 <= value <= 1.0, (
            f"Value for '{key}' is out of range [0.0, 1.0]: {value}"
        )
