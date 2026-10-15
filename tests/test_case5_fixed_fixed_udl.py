"""Phase B, Case 5: beam fixed at both ends with a full-length uniform load (Spec)."""

import numpy as np
import pytest

from solver import Beam, Support, UniformLoad, solve_beam

# Spec, Case 5: fixed at x = 0 and x = 220 mm; uniform load 0.05 N/mm
# downward from x = 0 to 220 mm (solver input w = -0.05); measured test bar.
CASE_5 = Beam(L=220.0, E=68900.0, I=65.68,
              supports=[Support(0.0, "fixed"), Support(220.0, "fixed")],
              uniform_loads=[UniformLoad(0.0, 220.0, -0.05)])

# Spec tolerances: relative 1e-6; for expected values of exactly 0,
# absolute 1e-9 N for shear and reaction forces, and 1e-6 N*mm for
# moments and reaction moments. theta(0) and theta(220) are 0 exactly.
REL_TOL = 1e-6
ZERO_TOL_V = 1e-9

# The required nodes are only x = 0 and 220, so the beam is one segment
# and n elements; n = 1 has no node at x = 110.
ALL_N = [1, 2, 4]
N_WITH_MIDSPAN_NODE = [2, 4]


def node(result, x):
    """Index of the mesh node at exactly x (mm); subdivision places it exactly."""
    (i,) = np.flatnonzero(result.x_nodes == x)
    return i


@pytest.mark.parametrize("n", ALL_N)
def test_end_moments(n):
    """M(0) = M(220) = -q L^2 / 12 = -201.666667 N*mm (hogging; exact value -605/3)."""
    result = solve_beam(CASE_5, n)
    assert (result.x_vm[0], result.x_vm[-1]) == (0.0, 220.0)
    assert result.M[0] == pytest.approx(-201.666667, rel=REL_TOL, abs=0)
    assert result.M[-1] == pytest.approx(-201.666667, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", ALL_N)
def test_end_shears(n):
    """V(0) = +q L / 2 = +5.5 N; V at x = 220 (just-left value) = -5.5 N."""
    result = solve_beam(CASE_5, n)
    assert result.V[0] == pytest.approx(5.5, rel=REL_TOL, abs=0)
    assert result.V[-1] == pytest.approx(-5.5, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", ALL_N)
def test_end_rotations_held(n):
    """theta(0) = theta(220) = 0 rad exactly (held by the supports)."""
    result = solve_beam(CASE_5, n)
    assert result.theta[0] == 0.0
    assert result.theta[-1] == 0.0


@pytest.mark.parametrize("n", ALL_N)
def test_reactions(n):
    """Solver convention: force +q L / 2 = +5.5 N at both walls; moment +q L^2 / 12 =
    +201.666667 N*mm at x = 0 (counterclockwise), -201.666667 N*mm at x = 220 (clockwise).
    """
    left, right = solve_beam(CASE_5, n).reactions
    assert (left.x, right.x) == (0.0, 220.0)
    assert left.force == pytest.approx(5.5, rel=REL_TOL, abs=0)
    assert right.force == pytest.approx(5.5, rel=REL_TOL, abs=0)
    assert left.moment == pytest.approx(201.666667, rel=REL_TOL, abs=0)
    assert right.moment == pytest.approx(-201.666667, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_WITH_MIDSPAN_NODE)
def test_midspan_deflection(n):
    """v(110) = -0.06740268 mm (|v| = q L^4 / 384EI)."""
    result = solve_beam(CASE_5, n)
    assert result.v[node(result, 110.0)] == pytest.approx(-0.06740268, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", N_WITH_MIDSPAN_NODE)
def test_midspan_moment_and_shear(n):
    """M(110) = +q L^2 / 24 = +100.833333 N*mm (exact value 605/6) and V(110) = 0 N,
    just left and just right of x = 110.
    """
    result = solve_beam(CASE_5, n)
    at_mid = np.flatnonzero(result.x_vm == 110.0)
    assert len(at_mid) == 2  # the value just left of the node, then just right
    for i in at_mid:
        assert result.M[i] == pytest.approx(100.833333, rel=REL_TOL, abs=0)
        assert result.V[i] == pytest.approx(0.0, abs=ZERO_TOL_V)


def test_no_free_unknowns_at_n1():
    """n = 1: no unknown is free, the solve is skipped, and the scaled condition number is NaN."""
    assert np.isnan(solve_beam(CASE_5, 1).cond)
