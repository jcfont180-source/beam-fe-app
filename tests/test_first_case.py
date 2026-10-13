"""Checkpoint A4 gate: first test case (Decision 4) and the cantilever reactions."""

import numpy as np
import pytest

from solver import Beam, PointLoad, Support, solve_beam

# Spec, "First test case (Decision 4)": cantilever fixed at x = 0, point
# load at x = L, measured test bar, P = 4.905 N downward.
CANTILEVER = Beam(L=220.0, E=68900.0, I=65.68,
                  supports=[Support(0.0, "fixed")],
                  point_loads=[PointLoad(220.0, -4.905)])
REL_TOL = 1e-6


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tip_deflection(n):
    """v(L) = -3.847100 mm (|v| = P L^3 / 3EI)."""
    assert solve_beam(CANTILEVER, n).v[-1] == pytest.approx(-3.847100, rel=REL_TOL)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tip_rotation(n):
    """theta(L) = -0.02623022 rad (|theta| = P L^2 / 2EI)."""
    assert solve_beam(CANTILEVER, n).theta[-1] == pytest.approx(-0.02623022, rel=REL_TOL)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fixed_end_moment(n):
    """M(0) = -1079.1 N*mm (|M| = P L, hogging), textbook convention."""
    result = solve_beam(CANTILEVER, n)
    assert result.x_vm[0] == 0.0
    assert result.M[0] == pytest.approx(-1079.1, rel=REL_TOL)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fixed_end_reactions(n):
    """Reactions at x = 0, solver convention: force +P = +4.905 N, moment +P L = +1079.1 N*mm."""
    (reaction,) = solve_beam(CANTILEVER, n).reactions
    assert reaction.x == 0.0
    assert reaction.force == pytest.approx(4.905, rel=REL_TOL)
    assert reaction.moment == pytest.approx(1079.1, rel=REL_TOL)


def test_shear_along_beam():
    """V = +P = +4.905 N at every reported point (n = 4), textbook convention."""
    np.testing.assert_allclose(solve_beam(CANTILEVER, 4).V, 4.905, rtol=REL_TOL)


def test_moment_along_beam():
    """M(x) = -P (L - x) = -4.905 (220 - x) N*mm at every reported point (n = 4)."""
    result = solve_beam(CANTILEVER, 4)
    np.testing.assert_allclose(result.x_vm, [0, 55, 55, 110, 110, 165, 165, 220],
                               rtol=0, atol=1e-12)
    expected = [-1079.1, -809.325, -809.325, -539.55, -539.55, -269.775, -269.775, 0.0]
    # Absolute tolerance, for the zero at the free end: 1e-6 of |M(0)|.
    np.testing.assert_allclose(result.M, expected, rtol=REL_TOL, atol=REL_TOL * 1079.1)
