"""Phase B, Case 4: simply supported beam with a full-length uniform load (Spec)."""

import numpy as np
import pytest

from solver import Beam, Support, UniformLoad, solve_beam

# Spec, Case 4: pin at x = 0, roller at x = 220 mm; uniform load 0.05 N/mm
# downward from x = 0 to 220 mm (solver input w = -0.05); measured test bar.
CASE_4 = Beam(L=220.0, E=68900.0, I=65.68,
              supports=[Support(0.0, "pin"), Support(220.0, "roller")],
              uniform_loads=[UniformLoad(0.0, 220.0, -0.05)])

# Spec tolerances: relative 1e-6; for expected values of exactly 0,
# absolute 1e-9 N for shear and reaction forces, and 1e-6 N*mm for
# moments and reaction moments.
REL_TOL = 1e-6
ZERO_TOL_V = 1e-9
ZERO_TOL_M = 1e-6

# The required nodes are only x = 0 and 220, so the beam is one segment
# and n elements; n = 1 has no node at x = 110.
ALL_N = [1, 2, 4]
N_WITH_MIDSPAN_NODE = [2, 4]


def node(result, x):
    """Index of the mesh node at exactly x (mm); subdivision places it exactly."""
    (i,) = np.flatnonzero(result.x_nodes == x)
    return i


@pytest.mark.parametrize("n", ALL_N)
def test_end_rotations(n):
    """theta(0) = -0.004902013 rad, theta(220) = +0.004902013 rad (|theta| = q L^3 / 24EI)."""
    result = solve_beam(CASE_4, n)
    assert result.theta[0] == pytest.approx(-0.004902013, rel=REL_TOL, abs=0)
    assert result.theta[-1] == pytest.approx(0.004902013, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", ALL_N)
def test_end_moments(n):
    """M(0) = 0 N*mm and M(220) = 0 N*mm at the pin and the roller."""
    result = solve_beam(CASE_4, n)
    assert (result.x_vm[0], result.x_vm[-1]) == (0.0, 220.0)
    assert result.M[0] == pytest.approx(0.0, abs=ZERO_TOL_M)
    assert result.M[-1] == pytest.approx(0.0, abs=ZERO_TOL_M)


@pytest.mark.parametrize("n", ALL_N)
def test_end_shears(n):
    """V(0) = +q L / 2 = +5.5 N; V at x = 220 (just-left value) = -5.5 N (V = q L / 2 - q x)."""
    result = solve_beam(CASE_4, n)
    assert result.V[0] == pytest.approx(5.5, rel=REL_TOL, abs=0)
    assert result.V[-1] == pytest.approx(-5.5, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", ALL_N)
def test_reactions(n):
    """Force +q L / 2 = +5.5 N at x = 0 and x = 220; moment 0 at both (pin and roller)."""
    pin, roller = solve_beam(CASE_4, n).reactions
    assert (pin.x, roller.x) == (0.0, 220.0)
    for reaction in (pin, roller):
        assert reaction.force == pytest.approx(5.5, rel=REL_TOL, abs=0)
        assert reaction.moment == pytest.approx(0.0, abs=ZERO_TOL_M)


@pytest.mark.parametrize("n", N_WITH_MIDSPAN_NODE)
def test_midspan_deflection(n):
    """v(110) = -0.3370134 mm (|v| = 5 q L^4 / 384EI)."""
    result = solve_beam(CASE_4, n)
    assert result.v[node(result, 110.0)] == pytest.approx(-0.3370134, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_WITH_MIDSPAN_NODE)
def test_midspan_moment_and_shear(n):
    """M(110) = +q L^2 / 8 = +302.5 N*mm and V(110) = 0 N, just left and just right of x = 110."""
    result = solve_beam(CASE_4, n)
    at_mid = np.flatnonzero(result.x_vm == 110.0)
    assert len(at_mid) == 2  # the value just left of the node, then just right
    for i in at_mid:
        assert result.M[i] == pytest.approx(302.5, rel=REL_TOL, abs=0)
        assert result.V[i] == pytest.approx(0.0, abs=ZERO_TOL_V)
