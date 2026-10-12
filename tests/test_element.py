"""Checkpoint A1 tests: beam element stiffness and uniform-load vector."""

import numpy as np
import pytest

from solver.element import element_stiffness, uniform_load_vector

# (EI in N*mm^2, L in mm): unit values, and the measured test bar
# (E = 68,900 MPa, I = 65.68 mm^4) with a 55 mm element.
STIFFNESS_CASES = [
    pytest.param(1.0, 1.0, id="unit"),
    pytest.param(68900 * 65.68, 55.0, id="test-bar"),
]


def k_hat(L):
    """k / (EI/L^3), as printed by docs/derivation/beam_element_stiffness.m."""
    return np.array([
        [12, 6 * L, -12, 6 * L],
        [6 * L, 4 * L**2, -6 * L, 2 * L**2],
        [-12, -6 * L, 12, -6 * L],
        [6 * L, 2 * L**2, -6 * L, 4 * L**2],
    ])


def assert_zero(residual, k, L):
    """Assert residual = 0, up to floating-point noise relative to the size of k."""
    np.testing.assert_allclose(residual, 0.0, atol=1e-12 * np.abs(k).max() * max(1.0, L))


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_stiffness_matches_matlab_derivation(EI, L):
    """k = (EI/L^3) * k_hat, with k_hat from beam_element_stiffness.m."""
    np.testing.assert_allclose(element_stiffness(EI, L), EI / L**3 * k_hat(L), rtol=1e-12)


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_stiffness_symmetry(EI, L):
    """Symmetry: k - k^T = 0."""
    k = element_stiffness(EI, L)
    assert_zero(k - k.T, k, L)


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_stiffness_vertical_equilibrium(EI, L):
    """Vertical equilibrium: row 1 + row 3 = 0."""
    k = element_stiffness(EI, L)
    assert_zero(k[0] + k[2], k, L)


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_stiffness_rigid_translation(EI, L):
    """Rigid translation: k * [1, 0, 1, 0] = 0."""
    k = element_stiffness(EI, L)
    assert_zero(k @ [1.0, 0.0, 1.0, 0.0], k, L)


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_stiffness_rigid_rotation(EI, L):
    """Rigid rotation: k * [0, 1, L, 1] = 0."""
    k = element_stiffness(EI, L)
    assert_zero(k @ [0.0, 1.0, L, 1.0], k, L)


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_stiffness_moment_equilibrium(EI, L):
    """Moment equilibrium about node 1: row 2 + row 4 + L * row 3 = 0."""
    k = element_stiffness(EI, L)
    assert_zero(k[1] + k[3] + L * k[2], k, L)


# Uniform-load cases (w in N/mm, Le in mm), chosen so the expected vector
# [w*Le/2, w*Le^2/12, w*Le/2, -w*Le^2/12] is easy to check by hand.
UNIFORM_LOAD_CASES = [
    (-0.6, 10.0, [-3.0, -5.0, -3.0, 5.0]),  # downward load
    (2.4, 5.0, [6.0, 5.0, 6.0, -5.0]),  # upward load
]


@pytest.mark.parametrize("w, Le, expected", UNIFORM_LOAD_CASES)
def test_uniform_load_vector(w, Le, expected):
    """f_eq = [w*Le/2, w*Le^2/12, w*Le/2, -w*Le^2/12]."""
    np.testing.assert_allclose(uniform_load_vector(w, Le), expected, rtol=1e-12)


@pytest.mark.parametrize("w, Le", [(w, Le) for w, Le, _ in UNIFORM_LOAD_CASES])
def test_uniform_load_force_sum(w, Le):
    """Vertical equilibrium: F1 + F2 = w*Le (the total load on the element)."""
    f = uniform_load_vector(w, Le)
    assert f[0] + f[2] == pytest.approx(w * Le, rel=1e-12)
