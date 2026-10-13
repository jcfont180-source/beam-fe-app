"""Checkpoint A4 gate: stability check, must-fail and must-pass tests."""

import numpy as np
import pytest

from solver import Beam, PointLoad, Support, UnstableBeamError, solve_beam
from solver.solve import assemble_stiffness, constrained_dofs, scaled_condition_number

# Copied from the Spec rather than imported from the solver, so a typo in
# the solver's copy makes the must-fail tests fail.
SPEC_MESSAGE = ("Beam is unstable: add supports. The current supports do not "
                "prevent rigid-body motion.")

E, I, L, P = 68900.0, 65.68, 220.0, 4.905  # measured test bar, 500 g load


def bar_beam(supports, load_x):
    """Test-bar beam with the given supports and P = 4.905 N downward at load_x (mm)."""
    return Beam(L=L, E=E, I=I, supports=supports, point_loads=[PointLoad(load_x, -P)])


def assert_finite(result):
    """Every output of a solved beam, including cond, is a finite number."""
    for values in [result.x_nodes, result.v, result.theta, result.x_vm, result.V, result.M,
                   [r.force for r in result.reactions], [r.moment for r in result.reactions],
                   [result.cond]]:
        assert np.all(np.isfinite(values))


# Must-fail tests. The load and n do not matter: the check runs before meshing.
@pytest.mark.parametrize("supports", [
    pytest.param([], id="no-supports"),
    pytest.param([Support(0.0, "pin")], id="pin-at-0-only"),
])
def test_must_fail(supports):
    """Unstable supports raise UnstableBeamError with exactly the Spec's message."""
    with pytest.raises(UnstableBeamError) as excinfo:
        solve_beam(bar_beam(supports, L), 4)
    assert str(excinfo.value) == SPEC_MESSAGE


@pytest.mark.parametrize("n", [1, 2, 4])
def test_must_pass_cantilever(n):
    """Cantilever fixed at x = 0 (load at x = L) solves without error."""
    assert_finite(solve_beam(bar_beam([Support(0.0, "fixed")], L), n))


@pytest.mark.parametrize("n", [1, 2, 4])
def test_must_pass_pin_roller(n):
    """Pin at x = 0 + roller at x = L (load at midspan) solves without error."""
    assert_finite(solve_beam(bar_beam([Support(0.0, "pin"), Support(L, "roller")], L / 2), n))


def test_must_pass_regression_load_near_pin():
    """Regression: pin at 0 + roller at L, load at x = 0.1 mm, n = 50 solves without error.

    The old rule, raw cond(K_reduced) > 1e14, wrongly rejected this stable beam.
    """
    assert_finite(solve_beam(bar_beam([Support(0.0, "pin"), Support(L, "roller")], 0.1), 50))


def test_fully_restrained_beam():
    """Fixed at both ends, n = 1, no loads: nothing can move, so v = theta = 0 and cond = NaN."""
    result = solve_beam(Beam(L=L, E=E, I=I,
                             supports=[Support(0.0, "fixed"), Support(L, "fixed")]), 1)
    assert np.all(result.v == 0.0)
    assert np.all(result.theta == 0.0)
    assert np.isnan(result.cond)


def test_scaled_condition_number_of_diagonal_matrix():
    """cond_2(D^-1/2 K D^-1/2) = 1 for K = diag(4, 1); unscaled, cond would be 4."""
    assert scaled_condition_number(np.diag([4.0, 1.0])) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("supports, size", [
    pytest.param([Support(0.0, "fixed")], 4, id="cantilever"),
    pytest.param([Support(0.0, "pin")], 5, id="pin-node-1-only"),
    pytest.param([Support(0.0, "pin"), Support(2.0, "roller")], 4, id="pin-1-roller-3"),
])
def test_support_case_ranks_match_matlab(supports, size):
    """Section 7 of global_stiffness_assembly.m: reduced K is size x size with rank 4.

    Two unit elements (EI = 1, L = 1, nodes at 0, 1, 2); the rows and columns
    held by the supports (from constrained_dofs) are removed.
    """
    x_nodes = np.array([0.0, 1.0, 2.0])
    held = constrained_dofs(Beam(L=2.0, E=1.0, I=1.0, supports=supports), x_nodes)
    free = [i for i in range(6) if i not in held]
    K_ff = assemble_stiffness(x_nodes, 1.0)[np.ix_(free, free)]
    assert K_ff.shape == (size, size)
    assert np.linalg.matrix_rank(K_ff) == 4
