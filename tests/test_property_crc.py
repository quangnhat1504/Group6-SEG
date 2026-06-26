"""Property-based tests for CRC calibration guarantee.

Feature: seg-experimental-validation, Property 5: CRC Calibration Guarantee

Validates: Requirements 3.5, 4.4
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to path so we can import crc_threshold
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_conformal_rerank import crc_threshold, LOSS_BOUND  # noqa: E402

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite


@composite
def crc_inputs(draw):
    """Generate valid CRC inputs: signal/loss dicts, cal_ids subset, and alpha.

    - n_queries: 10-100 query IDs
    - signal dict: query_id -> float in [0, 5]
    - loss dict: query_id -> float in [0, 1]
    - cal_ids: a non-empty subset of at least 5 query IDs
    - alpha: constrained so that the CRC guarantee is mathematically achievable

    The CRC guarantee (n * Rhat + B) / (n + 1) <= alpha has a minimum floor:
    even when Rhat = 0, it requires alpha >= B / (n + 1). We constrain alpha
    accordingly so the test only checks cases where the guarantee is achievable.
    """
    n_queries = draw(st.integers(min_value=10, max_value=100))
    query_ids = [f"q{i}" for i in range(n_queries)]

    signal = {
        qid: draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
        for qid in query_ids
    }
    loss = {
        qid: draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for qid in query_ids
    }

    # Select a non-empty subset of at least 5 query IDs as calibration set
    n_cal = draw(st.integers(min_value=5, max_value=n_queries))
    cal_ids = draw(
        st.lists(
            st.sampled_from(query_ids),
            min_size=n_cal,
            max_size=n_cal,
            unique=True,
        )
    )

    # Alpha must be at least B/(n_cal+1) for the guarantee to be achievable
    # (even with Rhat=0, (n*0 + B)/(n+1) = B/(n+1) must be <= alpha)
    B = LOSS_BOUND  # 1.0
    alpha_min = B / (n_cal + 1) + 1e-9  # slightly above the floor
    alpha_max = 0.5
    # Ensure alpha_min < alpha_max (always true since n_cal >= 5 → floor <= 1/6 < 0.5)
    alpha = draw(st.floats(min_value=alpha_min, max_value=alpha_max, allow_nan=False, allow_infinity=False))

    return signal, loss, cal_ids, alpha


@settings(max_examples=100)
@given(data=crc_inputs())
def test_crc_calibration_guarantee(data):
    """Property 5: CRC Calibration Guarantee.

    For any set of calibration query IDs (size n >= 1), a signal dict and loss dict
    covering those IDs (with losses in [0, 1]), and alpha in (0, 1], the lambda
    returned by crc_threshold SHALL satisfy the CRC calibration condition:
        (n * Rhat(lambda) + B) / (n + 1) <= alpha
    where Rhat is the mean loss of queries skipped (signal > lambda) on the
    calibration set and B = 1.

    **Validates: Requirements 3.5, 4.4**
    """
    signal, loss, cal_ids, alpha = data
    B = LOSS_BOUND  # 1.0

    lam = crc_threshold(signal, loss, cal_ids, alpha, B=B)

    # Compute Rhat for the returned lambda
    n = len(cal_ids)
    skipped = [q for q in cal_ids if signal[q] > lam]
    rhat = sum(loss[q] for q in skipped) / n

    # CRC calibration guarantee must hold
    assert (n * rhat + B) / (n + 1) <= alpha, (
        f"CRC guarantee violated: (n={n} * rhat={rhat:.6f} + B={B}) / (n+1={n+1}) "
        f"= {(n * rhat + B) / (n + 1):.6f} > alpha={alpha}"
    )
