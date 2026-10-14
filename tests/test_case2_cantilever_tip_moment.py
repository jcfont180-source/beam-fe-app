"""Phase B, Case 2: cantilever with an applied tip moment (Spec)."""

import numpy as np
import pytest

from solver import AppliedMoment, Beam, Support, solve_beam

# Spec, Case 2: fixed at x = 0; applied moment M0 = 500 N*mm counterclockwise
# at x = 220 mm (solver input +500); no other loads; measured test bar.
CASE_2 = Beam(L=220.0, E=68900.0, I=65.68,
              supports=[Support(0.0, "fixed")],
              moments=[AppliedMoment(220.0, 500.0)])

# Spec tolerances: relative 1e-6; for expected values of exactly 0,
# absolute 1e-9 N for shear, 1e-6 N*mm for moment, and 1e-9 N for the
# zero reaction force.
REL_TOL = 1e-6
ZERO_TOL_V = 1e-9
ZERO_TOL_REACTION = 1e-9


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tip_deflection(n):
    """v(L) = +2.673825 mm (|v| = M0 L^2 / 2EI): the beam bends into a smile."""
    assert solve_beam(CASE_2, n).v[-1] == pytest.approx(2.673825, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tip_rotation(n):
    """theta(L) = +0.02430750 rad (|theta| = M0 L / EI)."""
    assert solve_beam(CASE_2, n).theta[-1] == pytest.approx(0.02430750, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fixed_end_moment_and_shear(n):
    """M(0) = +M0 = +500 N*mm and V(0) = 0 N, textbook (diagram) convention."""
    result = solve_beam(CASE_2, n)
    assert result.x_vm[0] == 0.0
    assert result.M[0] == pytest.approx(500.0, rel=REL_TOL, abs=0)
    assert result.V[0] == pytest.approx(0.0, abs=ZERO_TOL_V)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fixed_end_reaction(n):
    """Reaction at x = 0, solver convention: force 0 N, moment -500 N*mm (wall twists clockwise).

    Opposite in sign to the diagram moment M(0) = +500 N*mm at the same point.
    """
    (reaction,) = solve_beam(CASE_2, n).reactions
    assert reaction.x == 0.0
    assert reaction.force == pytest.approx(0.0, abs=ZERO_TOL_REACTION)
    assert reaction.moment == pytest.approx(-500.0, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("x", [55.0, 110.0, 165.0])
def test_interior_node_values(x):
    """n = 4: V = 0 N and M = +M0 = +500 N*mm, just left and just right of node x."""
    result = solve_beam(CASE_2, 4)
    at_x = np.flatnonzero(result.x_vm == x)
    assert len(at_x) == 2  # the value just left of the node, then just right
    for i in at_x:
        assert result.V[i] == pytest.approx(0.0, abs=ZERO_TOL_V)
        assert result.M[i] == pytest.approx(500.0, rel=REL_TOL, abs=0)


def test_free_end_values():
    """n = 4, x = 220 (free end, just-left value only): V = 0 N, M = +500 N*mm.

    M is not 0 here because the moment is applied at the tip.
    """
    result = solve_beam(CASE_2, 4)
    assert result.x_vm[-1] == 220.0
    assert np.count_nonzero(result.x_vm == 220.0) == 1
    assert result.V[-1] == pytest.approx(0.0, abs=ZERO_TOL_V)
    assert result.M[-1] == pytest.approx(500.0, rel=REL_TOL, abs=0)
