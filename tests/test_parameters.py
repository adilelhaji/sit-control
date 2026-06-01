"""Unit tests for the parameters module."""

from __future__ import annotations

import pytest

from sit_control.parameters import (
    K_ALMEIDA_2022,
    K_JOIN_2026,
    BiologicalParameters,
)


def test_default_preset_reproduces_almeida_F_bar() -> None:
    """The default constructor must reproduce F_bar = 11037 (Almeida 2022).

    Almeida et al. (2022) Tables 2, 3 and 4 explicitly report
    F_bar = 11037 with nu_E = 0.05. This test asserts that the default
    K and the persistence-equilibrium formula are jointly consistent
    with that reference value.
    """
    params = BiologicalParameters()
    assert params.K == pytest.approx(K_ALMEIDA_2022)
    assert params.F_bar == pytest.approx(11037.0, abs=2.0)


def test_almeida2022_preset_F_bar() -> None:
    """The almeida2022() preset must reproduce F_bar = 11037 exactly."""
    params = BiologicalParameters.almeida2022()
    assert params.F_bar == pytest.approx(11037.0, abs=2.0)


def test_join2026_preset_K() -> None:
    """The join2026() preset must set K = 22200 (Join 2026, Sec. 5.2)."""
    params = BiologicalParameters.join2026()
    assert params.K == pytest.approx(K_JOIN_2026)


def test_presets_share_biological_parameters() -> None:
    """Both presets must share every parameter except K.

    K is the only site-specific calibration parameter (Strugarek 2019,
    Sec. 4.1); all other parameters are biological constants.
    """
    a = BiologicalParameters.almeida2022()
    j = BiologicalParameters.join2026()
    for field in ("beta_E", "delta_E", "delta_M", "delta_F",
                  "delta_s", "nu_E", "nu", "gamma_s"):
        assert getattr(a, field) == getattr(j, field), (
            f"{field} differs between presets: "
            f"almeida={getattr(a, field)} join={getattr(j, field)}"
        )
    assert a.K != j.K


def test_R0_independent_of_K() -> None:
    """R0 must not depend on K (it depends only on demographic rates)."""
    a = BiologicalParameters.almeida2022()
    j = BiologicalParameters.join2026()
    assert a.R0 == pytest.approx(j.R0, rel=1e-12)


def test_F_bar_scales_linearly_with_K() -> None:
    """F_bar must scale linearly with K (other parameters fixed)."""
    a = BiologicalParameters.almeida2022()
    j = BiologicalParameters.join2026()
    ratio_K = j.K / a.K
    ratio_F = j.F_bar / a.F_bar
    assert ratio_F == pytest.approx(ratio_K, rel=1e-12)


def test_override_via_preset() -> None:
    """Presets must accept field overrides without breaking K."""
    p = BiologicalParameters.almeida2022(nu_E=0.1)
    assert p.K == pytest.approx(K_ALMEIDA_2022)
    assert p.nu_E == pytest.approx(0.1)


def test_negative_parameter_rejected() -> None:
    """Construction with non-positive parameters must raise ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        BiologicalParameters(K=-1.0)


def test_nu_out_of_range_rejected() -> None:
    """nu must lie in the open interval (0, 1)."""
    with pytest.raises(ValueError, match="nu must be in"):
        BiologicalParameters(nu=1.5)


def test_gamma_s_out_of_range_rejected() -> None:
    """gamma_s must lie in (0, 1]: a sterile male cannot be more competitive
    than a wild one in this model (Ch. 2)."""
    with pytest.raises(ValueError, match=r"gamma_s must be in \(0, 1\]"):
        BiologicalParameters(gamma_s=1.5)
