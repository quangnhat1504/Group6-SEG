"""Property-based tests for paired bootstrap output invariants.

Feature: seg-experimental-validation, Property 1: Paired Bootstrap Output Invariants

Validates: Requirements 1.1, 1.2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts/ to path so we can import paired_bootstrap
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_significance_conformal import paired_bootstrap  # noqa: E402


@settings(max_examples=100)
@given(
    sys_list=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=200,
    ),
    base_list=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=200,
    ),
)
def test_paired_bootstrap_output_invariants(sys_list: list[float], base_list: list[float]) -> None:
    """Property 1: Paired Bootstrap Output Invariants.

    For any two arrays of per-query metric scores (system and baseline) of the
    same length n >= 2 with values in [0, 1], paired_bootstrap returns
    (mean_diff, ci_lo, ci_hi, p) where:
      - mean_diff equals the true observed mean of (system - baseline)
      - ci_lo <= mean_diff <= ci_hi (CI contains the point estimate)
      - 0 <= p <= 1

    **Validates: Requirements 1.1, 1.2**
    """
    # Make arrays the same length (use the shorter length)
    n = min(len(sys_list), len(base_list))
    sys_vals = np.array(sys_list[:n], dtype=np.float64)
    base_vals = np.array(base_list[:n], dtype=np.float64)

    # Use a fixed rng seed for reproducibility
    rng = np.random.default_rng(42)

    mean_diff, ci_lo, ci_hi, p_value = paired_bootstrap(sys_vals, base_vals, rng)

    # Property 1a: mean_diff equals the true observed mean of (system - baseline)
    expected_mean = float(np.mean(sys_vals - base_vals))
    assert np.isclose(mean_diff, expected_mean, atol=1e-10), (
        f"mean_diff={mean_diff} != expected={expected_mean}"
    )

    # Property 1b: ci_lo <= mean_diff <= ci_hi (CI contains point estimate)
    assert ci_lo <= mean_diff <= ci_hi, (
        f"CI [{ci_lo}, {ci_hi}] does not contain mean_diff={mean_diff}"
    )

    # Property 1c: 0 <= p <= 1
    assert 0.0 <= p_value <= 1.0, f"p_value={p_value} not in [0, 1]"
