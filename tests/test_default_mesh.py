"""Phase B, Default mesh: n = 20 per segment, or the largest n that fits the cap (Spec)."""

from dataclasses import fields

import numpy as np
import pytest

from solver import Beam, PointLoad, Support, UniformLoad, solve_beam

E, I, L = 68900.0, 65.68, 220.0  # measured test bar

# Case 4's beam: pin at x = 0, roller at x = 220 mm, w = -0.05 N/mm over the span.
CASE_4_BEAM = Beam(L=L, E=E, I=I,
                   supports=[Support(0.0, "pin"), Support(L, "roller")],
                   uniform_loads=[UniformLoad(0.0, L, -0.05)])

# 26 segments: pin at 0 and roller at 220, plus 25 point loads at
# x = 8, 16, ..., 200 mm, so the required nodes are 0, 8, ..., 200, 220.
BEAM_26_SEGMENTS = Beam(L=L, E=E, I=I,
                        supports=[Support(0.0, "pin"), Support(L, "roller")],
                        point_loads=[PointLoad(8.0 * k, -1.0) for k in range(1, 26)])


def assert_same_results(a, b):
    """Every field of two BeamResults is identical: the same calculation, so exact equality."""
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=f.name)
        else:
            assert x == y, f.name


def test_default_is_20_per_segment():
    """Case 4's beam with no n argument gives the same results as n = 20."""
    assert_same_results(solve_beam(CASE_4_BEAM), solve_beam(CASE_4_BEAM, 20))


def test_default_drops_to_largest_n_that_fits():
    """26 segments, no n: n = floor(500 / 26) = 19, so 494 elements (26 x 20 = 520 > 500)."""
    result = solve_beam(BEAM_26_SEGMENTS)
    assert len(result.x_nodes) - 1 == 494
    assert_same_results(result, solve_beam(BEAM_26_SEGMENTS, 19))


def test_explicit_n_is_never_changed():
    """An explicit n = 20 on 26 segments (520 elements) exceeds the cap: ValueError, as before."""
    with pytest.raises(ValueError):
        solve_beam(BEAM_26_SEGMENTS, 20)


def test_default_raises_when_even_n_1_exceeds_cap():
    """501 segments, no n: even n = 1 gives 501 elements > 500, so ValueError."""
    beam = Beam(L=L, E=E, I=I,
                supports=[Support(0.0, "pin"), Support(L, "roller")],
                point_loads=[PointLoad(L * k / 501, -1.0) for k in range(1, 501)])
    with pytest.raises(ValueError):
        solve_beam(beam)
