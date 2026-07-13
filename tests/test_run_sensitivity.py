"""Unit tests for run_sensitivity helpers (no GEKKO required)."""

from __future__ import annotations

import dataclasses

import pytest

from sit_control.parameters import BiologicalParameters


@pytest.fixture
def params() -> BiologicalParameters:
    return BiologicalParameters()


# ---------------------------------------------------------------------------
# _perturb
# ---------------------------------------------------------------------------


def test_perturb_increases_field(params: BiologicalParameters) -> None:
    """A +20 % perturbation on delta_F should increase the field by exactly 20 %."""
    from scripts.run_sensitivity import _perturb

    perturbed = _perturb(params, "delta_F", +0.20)
    assert perturbed.delta_F == pytest.approx(params.delta_F * 1.20, rel=1e-9)


def test_perturb_decreases_field(params: BiologicalParameters) -> None:
    """A -20 % perturbation on nu_E should decrease the field by exactly 20 %."""
    from scripts.run_sensitivity import _perturb

    perturbed = _perturb(params, "nu_E", -0.20)
    assert perturbed.nu_E == pytest.approx(params.nu_E * 0.80, rel=1e-9)


def test_perturb_leaves_other_fields_unchanged(params: BiologicalParameters) -> None:
    """Perturbing K must not affect any other parameter."""
    from scripts.run_sensitivity import _perturb

    perturbed = _perturb(params, "K", +0.20)
    assert perturbed.delta_F == pytest.approx(params.delta_F)
    assert perturbed.nu_E == pytest.approx(params.nu_E)
    assert perturbed.delta_M == pytest.approx(params.delta_M)


def test_perturb_zero_factor_is_identity(params: BiologicalParameters) -> None:
    """Factor=0 should return parameters identical to the original."""
    from scripts.run_sensitivity import _perturb

    p2 = _perturb(params, "delta_F", 0.0)
    assert p2.delta_F == pytest.approx(params.delta_F)


def test_perturb_returns_frozen_dataclass(params: BiologicalParameters) -> None:
    """_perturb should return a new immutable BiologicalParameters instance."""
    from scripts.run_sensitivity import _perturb

    perturbed = _perturb(params, "K", +0.10)
    assert isinstance(perturbed, BiologicalParameters)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        perturbed.K = 99.0  # type: ignore[misc]


def test_perturb_R0_decreases_with_higher_delta_F(params: BiologicalParameters) -> None:
    """R0 should decrease when delta_F (female mortality) is increased."""
    from scripts.run_sensitivity import _perturb

    p_high = _perturb(params, "delta_F", +0.20)
    assert p_high.R0 < params.R0


def test_perturb_R0_increases_with_higher_nu_E(params: BiologicalParameters) -> None:
    """R0 should increase when nu_E (emergence rate) is increased."""
    from scripts.run_sensitivity import _perturb

    p_high = _perturb(params, "nu_E", +0.20)
    assert p_high.R0 > params.R0


def test_perturb_K_does_not_affect_R0(params: BiologicalParameters) -> None:
    """R0 is independent of K (K only appears in the density-dependence term)."""
    from scripts.run_sensitivity import _perturb

    p_high = _perturb(params, "K", +0.20)
    assert pytest.approx(params.R0, rel=1e-9) == p_high.R0


# ---------------------------------------------------------------------------
# plot_tornado (smoke test — does not call GEKKO)
# ---------------------------------------------------------------------------


def test_plot_tornado_runs_without_error() -> None:
    """plot_tornado should produce a Figure without raising."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.run_sensitivity import plot_tornado

    # Build a minimal synthetic sensitivity dict
    synthetic = {
        "baseline": {"J": 1.46e5, "t_epsilon_days": 90.0},
        "perturbations": {
            "delta_F": {
                "label": r"$\delta_F$",
                "baseline_value": 0.04,
                "runs": {
                    "-20%": {"J": 1.60e5, "t_epsilon_days": 100.0, "delta_J_pct": +9.6},
                    "+20%": {"J": 1.35e5, "t_epsilon_days": 80.0, "delta_J_pct": -7.5},
                },
            },
            "nu_E": {
                "label": r"$\nu_E$",
                "baseline_value": 0.05,
                "runs": {
                    "-20%": {"J": 1.52e5, "t_epsilon_days": 95.0, "delta_J_pct": +4.1},
                    "+20%": {"J": 1.41e5, "t_epsilon_days": 88.0, "delta_J_pct": -3.4},
                },
            },
            "K": {
                "label": r"$K$",
                "baseline_value": 22200.0,
                "runs": {
                    "-20%": {"J": 1.20e5, "t_epsilon_days": 85.0, "delta_J_pct": -17.8},
                    "+20%": {
                        "J": 1.71e5,
                        "t_epsilon_days": 105.0,
                        "delta_J_pct": +17.1,
                    },
                },
            },
        },
    }

    fig = plot_tornado(synthetic, save_path=None)
    assert fig is not None
    plt.close(fig)


def test_plot_tornado_has_correct_y_tick_count() -> None:
    """The tornado chart should have one row per parameter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.run_sensitivity import plot_tornado

    synthetic = {
        "baseline": {"J": 1.46e5, "t_epsilon_days": 90.0},
        "perturbations": {
            "delta_F": {
                "label": r"$\delta_F$",
                "baseline_value": 0.04,
                "runs": {
                    "-20%": {"J": 1.60e5, "t_epsilon_days": 100.0, "delta_J_pct": +9.6},
                    "+20%": {"J": 1.35e5, "t_epsilon_days": 80.0, "delta_J_pct": -7.5},
                },
            },
            "K": {
                "label": r"$K$",
                "baseline_value": 22200.0,
                "runs": {
                    "-20%": {"J": 1.20e5, "t_epsilon_days": 85.0, "delta_J_pct": -17.8},
                    "+20%": {
                        "J": 1.71e5,
                        "t_epsilon_days": 105.0,
                        "delta_J_pct": +17.1,
                    },
                },
            },
        },
    }
    fig = plot_tornado(synthetic, save_path=None)
    ax = fig.axes[0]
    assert len(ax.get_yticks()) == 2  # one per parameter
    plt.close(fig)
