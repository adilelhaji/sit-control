"""Unit tests for the model module."""

from __future__ import annotations

import numpy as np
import pytest

from sit_control.model import recruitment, rhs_reduced
from sit_control.parameters import BiologicalParameters


@pytest.fixture
def almeida_params() -> BiologicalParameters:
    """Parameters from Almeida et al. (2022), Table 1."""
    return BiologicalParameters()


def test_R0_value(almeida_params: BiologicalParameters) -> None:
    """R0 should match the value reported in the TFG (~76.56)."""
    assert almeida_params.R0 == pytest.approx(76.5625, rel=1e-3)


def test_F_bar_value(almeida_params: BiologicalParameters) -> None:
    """Persistence equilibrium F_bar should be ~11037 (Almeida 2022)."""
    assert almeida_params.F_bar == pytest.approx(11037.0, rel=1e-2)


def test_recruitment_at_origin(almeida_params: BiologicalParameters) -> None:
    """f(0, 0) should be 0 by removable singularity."""
    result = recruitment(0.0, 0.0, almeida_params)
    assert result == pytest.approx(0.0, abs=1e-10)


def test_recruitment_equilibrium(
    almeida_params: BiologicalParameters,
) -> None:
    """f(F_bar, 0) should be 0 (steady state)."""
    F_bar = almeida_params.F_bar
    result = recruitment(F_bar, 0.0, almeida_params)
    assert result == pytest.approx(0.0, abs=1e-3)


def test_rhs_reduced_shape(almeida_params: BiologicalParameters) -> None:
    """RHS should return a 2-element array."""
    state = np.array([1000.0, 500.0])
    result = rhs_reduced(0.0, state, lambda t: 0.0, almeida_params)
    assert result.shape == (2,)


def test_invalid_nu_raises() -> None:
    """nu outside (0, 1) should raise ValueError."""
    with pytest.raises(ValueError, match="nu"):
        BiologicalParameters(nu=1.5)


def test_negative_K_raises() -> None:
    """Negative K should raise ValueError."""
    with pytest.raises(ValueError, match="K"):
        BiologicalParameters(K=-100.0)