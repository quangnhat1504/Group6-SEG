"""Property-based tests for QPP features completeness.

Feature: seg-experimental-validation, Property 3: QPP Features Completeness

Validates: Requirements 2.1
"""
from __future__ import annotations

import math
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

from seg_retrieval.qpp import qpp_features


@composite
def hit_lists(draw):
    """Generate valid hit lists: 1-100 items with unique doc_ids, scores in [0, 50]."""
    n = draw(st.integers(min_value=1, max_value=100))
    doc_ids = draw(
        st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    # Sort by score descending (typical hit list ordering)
    hits = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    return hits


@settings(max_examples=100)
@given(
    hits=hit_lists(),
    mu_corpus=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    prefix=st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=("L",))),
)
def test_qpp_features_completeness(hits, mu_corpus, prefix):
    """Property 3: QPP Features Completeness.

    For any valid hit list and finite corpus mean, qpp_features() returns exactly 5 keys
    with expected prefix pattern, each a finite float.

    **Validates: Requirements 2.1**
    """
    result = qpp_features(hits, mu_corpus, prefix)

    # Exactly 5 keys
    assert len(result) == 5, f"Expected 5 keys, got {len(result)}: {list(result.keys())}"

    # Expected keys with correct prefix pattern
    expected_suffixes = {"wig", "nqc", "std", "max", "gap"}
    expected_keys = {f"{prefix}_{suffix}" for suffix in expected_suffixes}
    assert set(result.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(result.keys())}"
    )

    # All values are finite floats
    for key, value in result.items():
        assert isinstance(value, float), f"Value for '{key}' is not a float: {type(value)}"
        assert math.isfinite(value), f"Value for '{key}' is not finite: {value}"
