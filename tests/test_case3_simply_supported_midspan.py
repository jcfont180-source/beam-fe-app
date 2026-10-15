"""Phase B, Case 3: simply supported beam with a midspan point load (Spec)."""

import numpy as np
import pytest

from solver import Beam, PointLoad, Support, solve_beam

# Spec, Case 3: pin at x = 0, roller at x = 220 mm; P = 4.905 N downward
# at x = 110 mm (solver input -4.905); measured test bar.
CASE_3 = Beam(L=220.0, E=68900.0, I=65.68,
              supports=[Support(0.0, "pin"), Support(220.0, "roller")],
              point_loads=[PointLoad(110.0, -4.905)])

# Spec tolerances: relative 1e-6; for expected values of exactly 0,
# absolute 1e-6 N*mm for moment, including the zero reaction moments.
REL_TOL = 1e-6
ZERO_TOL_M = 1e-6

# n is per segment: the load node splits the beam into two segments, so
# n = 1, 2, 4 gives 2, 4, 8 elements.
N_PER_SEGMENT = [1, 2, 4]


def node(result, x):
    """Index of the mesh node at exactly x (mm); required nodes are placed exactly."""
    (i,) = np.flatnonzero(result.x_nodes == x)
    return i


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_midspan_deflection(n):
    """v(110) = -0.2404437 mm (|v| = P L^3 / 48EI)."""
    result = solve_beam(CASE_3, n)
    assert result.v[node(result, 110.0)] == pytest.approx(-0.2404437, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_end_rotations(n):
    """theta(0) = -0.003278778 rad, theta(220) = +0.003278778 rad (|theta| = P L^2 / 16EI)."""
    result = solve_beam(CASE_3, n)
    assert result.theta[0] == pytest.approx(-0.003278778, rel=REL_TOL, abs=0)
    assert result.theta[-1] == pytest.approx(0.003278778, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_midspan_moment(n):
    """M(110) = +269.775 N*mm (|M| = P L / 4), both just left and just right of x = 110."""
    result = solve_beam(CASE_3, n)
    at_mid = np.flatnonzero(result.x_vm == 110.0)
    assert len(at_mid) == 2
    for i in at_mid:
        assert result.M[i] == pytest.approx(269.775, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_end_moments(n):
    """M(0) = 0 N*mm and M(220) = 0 N*mm at the pin and the roller."""
    result = solve_beam(CASE_3, n)
    assert (result.x_vm[0], result.x_vm[-1]) == (0.0, 220.0)
    assert result.M[0] == pytest.approx(0.0, abs=ZERO_TOL_M)
    assert result.M[-1] == pytest.approx(0.0, abs=ZERO_TOL_M)


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_shear_jump_at_load(n):
    """V just left of x = 110 is +2.4525 N and just right is -2.4525 N: the jump equals P."""
    result = solve_beam(CASE_3, n)
    left, right = np.flatnonzero(result.x_vm == 110.0)  # just left, then just right
    assert result.V[left] == pytest.approx(2.4525, rel=REL_TOL, abs=0)
    assert result.V[right] == pytest.approx(-2.4525, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_end_shears(n):
    """V(0) = +2.4525 N; V at x = 220 (just-left value) = -2.4525 N."""
    result = solve_beam(CASE_3, n)
    assert result.V[0] == pytest.approx(2.4525, rel=REL_TOL, abs=0)
    assert result.V[-1] == pytest.approx(-2.4525, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_PER_SEGMENT)
def test_reactions(n):
    """Force +P/2 = +2.4525 N at x = 0 and x = 220; moment 0 at both (pin and roller)."""
    pin, roller = solve_beam(CASE_3, n).reactions
    assert (pin.x, roller.x) == (0.0, 220.0)
    for reaction in (pin, roller):
        assert reaction.force == pytest.approx(2.4525, rel=REL_TOL, abs=0)
        assert reaction.moment == pytest.approx(0.0, abs=ZERO_TOL_M)
