"""Phase B, Case 1: cantilever with a full-length uniform load (Spec)."""

import numpy as np
import pytest

from solver import Beam, Support, UniformLoad, solve_beam

# Spec, Case 1: fixed at x = 0, uniform load 0.05 N/mm downward from
# x = 0 to 220 mm (solver input w = -0.05), measured test bar.
CASE_1 = Beam(L=220.0, E=68900.0, I=65.68,
              supports=[Support(0.0, "fixed")],
              uniform_loads=[UniformLoad(0.0, 220.0, -0.05)])

# Spec tolerances: relative 1e-6; for expected values of exactly 0,
# absolute 1e-9 N for shear and 1e-6 N*mm for moment.
REL_TOL = 1e-6
ZERO_TOL_V = 1e-9
ZERO_TOL_M = 1e-6


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tip_deflection(n):
    """v(L) = -3.235328 mm (|v| = q L^4 / 8EI)."""
    assert solve_beam(CASE_1, n).v[-1] == pytest.approx(-3.235328, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_tip_rotation(n):
    """theta(L) = -0.01960805 rad (|theta| = q L^3 / 6EI)."""
    assert solve_beam(CASE_1, n).theta[-1] == pytest.approx(-0.01960805, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_fixed_end_moment_and_shear(n):
    """M(0) = -q L^2 / 2 = -1210 N*mm and V(0) = +q L = +11 N, textbook convention."""
    result = solve_beam(CASE_1, n)
    assert result.x_vm[0] == 0.0
    assert result.M[0] == pytest.approx(-1210.0, rel=REL_TOL, abs=0)
    assert result.V[0] == pytest.approx(11.0, rel=REL_TOL, abs=0)


@pytest.mark.parametrize("x, V, M", [
    (55.0, 8.25, -680.625),
    (110.0, 5.5, -302.5),
    (165.0, 2.75, -75.625),
])
def test_interior_node_values(x, V, M):
    """n = 4: V = +q (L - x) and M = -q (L - x)^2 / 2, just left and just right of node x."""
    result = solve_beam(CASE_1, 4)
    at_x = np.flatnonzero(result.x_vm == x)
    assert len(at_x) == 2  # the value just left of the node, then just right
    for i in at_x:
        assert result.V[i] == pytest.approx(V, rel=REL_TOL, abs=0)
        assert result.M[i] == pytest.approx(M, rel=REL_TOL, abs=0)


def test_free_end_values():
    """n = 4, x = 220 (free end, just-left value only): V = 0 N, M = 0 N*mm."""
    result = solve_beam(CASE_1, 4)
    assert result.x_vm[-1] == 220.0
    assert np.count_nonzero(result.x_vm == 220.0) == 1
    assert result.V[-1] == pytest.approx(0.0, abs=ZERO_TOL_V)
    assert result.M[-1] == pytest.approx(0.0, abs=ZERO_TOL_M)
