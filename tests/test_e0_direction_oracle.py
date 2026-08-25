"""Independent tests for the audit-only E0 direction-game oracle."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e0_direction_oracle import (  # noqa: E402
    maximin_simplex_oracle,
    min_norm_simplex_oracle,
)
from reference_solvers import (  # noqa: E402
    maximin_reference,
    min_norm_reference,
)


IDENTITY_GRAM = np.eye(3, dtype=np.float64)
TWO_CLIENT_ASYMMETRIC_GRAM = np.array([
    [1.0, 0.2],
    [0.2, 0.7],
], dtype=np.float64)
HARNESS_GRAM = np.array([
    [1.0, 0.47694383644995036, -0.053896080749643874],
    [0.47694383644995036, 1.0, 0.42062224967668665],
    [-0.053896080749643874, 0.42062224967668665,
     0.9999999999999999],
], dtype=np.float64)


def _assert_simplex_result(result, gram):
    assert set(result) == {
        "weights", "objective", "simplex_residual", "constraint_violation",
    }
    weights = result["weights"]
    assert weights.dtype == np.float64
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-10)
    assert np.min(weights) >= -1e-10
    assert result["simplex_residual"] <= 1e-10
    assert result["constraint_violation"] <= 1e-10
    return weights


@pytest.mark.parametrize(
    "gram", [IDENTITY_GRAM, TWO_CLIENT_ASYMMETRIC_GRAM, HARNESS_GRAM])
def test_min_norm_matches_reference_on_nonsingular_grams(gram):
    result = min_norm_simplex_oracle(gram)
    weights = _assert_simplex_result(result, gram)
    expected_value, _ = min_norm_reference(gram)

    assert result["objective"] == pytest.approx(expected_value, abs=1e-10)
    assert result["objective"] == pytest.approx(
        float(weights @ gram @ weights), abs=1e-12)


@pytest.mark.parametrize(
    "gram", [IDENTITY_GRAM, TWO_CLIENT_ASYMMETRIC_GRAM, HARNESS_GRAM])
def test_maximin_matches_reference_on_nonsingular_grams(gram):
    result = maximin_simplex_oracle(gram)
    weights = _assert_simplex_result(result, gram)
    expected_value, _ = maximin_reference(gram)

    assert result["objective"] == pytest.approx(expected_value, abs=1e-10)
    assert result["objective"] == pytest.approx(
        float(np.min(gram @ weights)), abs=1e-12)


@pytest.mark.parametrize(
    "oracle", [min_norm_simplex_oracle, maximin_simplex_oracle])
def test_identity_has_uniform_optimum(oracle):
    result = oracle(IDENTITY_GRAM)

    assert result["weights"] == pytest.approx(np.full(3, 1.0 / 3.0),
                                               abs=1e-12)
    assert result["objective"] == pytest.approx(1.0 / 3.0, abs=1e-12)


@pytest.mark.parametrize(
    "oracle", [min_norm_simplex_oracle, maximin_simplex_oracle])
def test_fully_aligned_singular_gram_has_analytic_value(oracle):
    gram = np.ones((3, 3), dtype=np.float64)

    first = oracle(gram)
    second = oracle(gram)

    _assert_simplex_result(first, gram)
    assert first["objective"] == pytest.approx(1.0, abs=1e-12)
    assert first["weights"] == pytest.approx(second["weights"], abs=0.0)


@pytest.mark.parametrize(
    "oracle", [min_norm_simplex_oracle, maximin_simplex_oracle])
def test_antipodal_singular_gram_has_zero_at_equal_weights(oracle):
    gram = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)

    result = oracle(gram)

    _assert_simplex_result(result, gram)
    assert result["weights"] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert result["objective"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "gram", [
        np.array([1.0, 2.0]),
        np.ones((2, 3)),
        np.empty((0, 0)),
        np.array([[1.0, 0.1], [0.2, 1.0]]),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, np.inf], [np.inf, 1.0]]),
    ],
)
@pytest.mark.parametrize(
    "oracle", [min_norm_simplex_oracle, maximin_simplex_oracle])
def test_invalid_grams_are_rejected(oracle, gram):
    with pytest.raises(ValueError):
        oracle(gram)
