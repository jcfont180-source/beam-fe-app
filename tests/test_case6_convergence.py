"""Phase B, Case 6: convergence study of the plotted moment between nodes (Spec)."""

import numpy as np
import pytest

from solver import Beam, Support, UniformLoad, solve_beam

# Spec, Case 6: Case 4's beam, pin at x = 0 and roller at x = 220 mm, with a
# uniform load of 0.05 N/mm downward over the whole span (solver input
# w = -0.05); measured test bar.
L, q = 220.0, 0.05
CASE_4_BEAM = Beam(L=L, E=68900.0, I=65.68,
                   supports=[Support(0.0, "pin"), Support(L, "roller")],
                   uniform_loads=[UniformLoad(0.0, L, -q)])
REL_TOL = 1e-6


def exact_moment(x):
    """M(x) = q x (L - x) / 2, the exact moment (sagging positive)."""
    return q * x * (L - x) / 2


@pytest.mark.parametrize("n, gap", [
    (2, 75.625),
    (4, 18.90625),
    (8, 4.7265625),
    (16, 1.181640625),
])
def test_plotted_moment_gap_at_element_midpoints(n, gap):
    """At every element midpoint, M(x_mid) - (M_left + M_right) / 2 = q h^2 / 8, h = 220/n."""
    result = solve_beam(CASE_4_BEAM, n)
    x_left, x_right = result.x_vm[0::2], result.x_vm[1::2]  # each element's two ends
    M_left, M_right = result.M[0::2], result.M[1::2]
    gaps = exact_moment((x_left + x_right) / 2) - (M_left + M_right) / 2
    assert len(gaps) == n
    np.testing.assert_allclose(gaps, gap, rtol=REL_TOL, atol=0)
