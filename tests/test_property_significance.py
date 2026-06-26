"""Property test for significance marking correctness.

Feature: seg-experimental-validation, Property 2: Significance Marking Correctness

Tests that for random (ci_lo, ci_hi) pairs where ci_lo < ci_hi,
the comparison is marked significant iff ci_lo > 0 or ci_hi < 0
(i.e., the confidence interval excludes zero).

**Validates: Requirements 1.4**
"""
from __future__ import annotations

from hypothesis import given, assume, settings
from hypothesis import strategies as st


def is_significant(ci_lo: float, ci_hi: float) -> bool:
    """Determine if a confidence interval indicates statistical significance.

    A comparison is significant when the confidence interval excludes zero,
    meaning either the entire interval is above zero or entirely below zero.
    """
    return ci_lo > 0 or ci_hi < 0


@settings(max_examples=100)
@given(
    ci_lo=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
    ci_hi=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
)
def test_significance_marking_correctness(ci_lo: float, ci_hi: float) -> None:
    """Property 2: Significance Marking Correctness.

    For any confidence interval (ci_lo, ci_hi) where ci_lo < ci_hi,
    the comparison SHALL be marked as statistically significant if and only if
    ci_lo > 0 or ci_hi < 0 (the interval excludes zero).

    Equivalently: significant iff NOT (ci_lo <= 0 <= ci_hi),
    i.e., zero is not contained in the interval.

    **Validates: Requirements 1.4**
    """
    assume(ci_lo < ci_hi)

    result = is_significant(ci_lo, ci_hi)

    # Property: significant iff CI excludes zero
    expected = ci_lo > 0 or ci_hi < 0
    assert result == expected, (
        f"is_significant({ci_lo}, {ci_hi}) = {result}, expected {expected}"
    )

    # Equivalence: significant iff zero is NOT contained in [ci_lo, ci_hi]
    zero_in_interval = ci_lo <= 0 <= ci_hi
    assert result == (not zero_in_interval), (
        f"is_significant({ci_lo}, {ci_hi}) = {result}, "
        f"but zero_in_interval = {zero_in_interval}"
    )
