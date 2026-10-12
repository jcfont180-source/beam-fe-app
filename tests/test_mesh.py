"""Checkpoint A2 tests: meshing rule (required nodes, merging, subdivision, limits)."""

from dataclasses import replace

import numpy as np
import pytest

from solver.mesh import build_mesh, node_index, required_nodes
from solver.model import AppliedMoment, Beam, PointLoad, Support, UniformLoad

E, I = 68900.0, 65.68  # test bar; the mesh does not depend on E or I

# First test case: cantilever fixed at x = 0, 4.905 N downward at x = L.
CANTILEVER = Beam(L=220.0, E=E, I=I,
                  supports=[Support(0.0, "fixed")],
                  point_loads=[PointLoad(220.0, -4.905)])

# One of every node trigger: pin at 0, roller at 200 (overhang to 220),
# point load at 50, applied moment at 120, uniform load from 80 to 160.
ALL_TRIGGERS = Beam(L=220.0, E=E, I=I,
                    supports=[Support(0.0, "pin"), Support(200.0, "roller")],
                    point_loads=[PointLoad(50.0, -4.905)],
                    moments=[AppliedMoment(120.0, 1000.0)],
                    uniform_loads=[UniformLoad(80.0, 160.0, -0.02)])


def loads_at(xs):
    """Point loads of -1 N at each location in xs (mm)."""
    return [PointLoad(x, -1.0) for x in xs]


def test_required_nodes_cantilever():
    """Rule 1: nodes at the beam ends 0 and L; the support and load sit on them."""
    np.testing.assert_array_equal(required_nodes(CANTILEVER), [0.0, 220.0])


def test_required_nodes_every_trigger():
    """Rule 1: ends, supports, point load, applied moment, and both uniform-load ends."""
    np.testing.assert_array_equal(required_nodes(ALL_TRIGGERS),
                                  [0.0, 50.0, 80.0, 120.0, 160.0, 200.0, 220.0])


def test_locations_within_tolerance_merge():
    """Rule 1 merge: locations 5e-10 mm apart (within 1e-9 mm) become one node."""
    beam = replace(CANTILEVER, point_loads=loads_at([100.0, 100.0 + 5e-10]))
    np.testing.assert_array_equal(required_nodes(beam), [0.0, 100.0, 220.0])


def test_locations_beyond_tolerance_stay_separate():
    """Rule 1 merge: locations 2e-9 mm apart (more than 1e-9 mm) stay separate nodes."""
    beam = replace(CANTILEVER, point_loads=loads_at([100.0, 100.0 + 2e-9]))
    np.testing.assert_array_equal(required_nodes(beam), [0.0, 100.0, 100.0 + 2e-9, 220.0])


def test_merging_keeps_beam_ends_exact():
    """Locations within 1e-9 mm of 0 or L merge into the ends, which stay exactly 0 and L."""
    beam = replace(CANTILEVER, point_loads=loads_at([5e-10, 220.0 - 5e-10]))
    np.testing.assert_array_equal(required_nodes(beam), [0.0, 220.0])


@pytest.mark.parametrize("n, expected", [
    (1, [0.0, 220.0]),
    (2, [0.0, 110.0, 220.0]),
    (4, [0.0, 55.0, 110.0, 165.0, 220.0]),
])
def test_subdivision_cantilever(n, expected):
    """Rule 3: x_k = a + k*(b - a)/n on the single segment from 0 to 220 mm."""
    np.testing.assert_allclose(build_mesh(CANTILEVER, n), expected, rtol=0, atol=1e-12)


def test_subdivision_unequal_segments():
    """Rule 3: each segment gets n equal elements of its own length (60/2 and 160/2)."""
    beam = Beam(L=220.0, E=E, I=I,
                supports=[Support(0.0, "pin"), Support(220.0, "roller")],
                point_loads=[PointLoad(60.0, -4.905)])
    np.testing.assert_allclose(build_mesh(beam, 2), [0.0, 30.0, 60.0, 140.0, 220.0],
                               rtol=0, atol=1e-12)


def test_subdivision_keeps_required_nodes_exact():
    """Rule 3: every required node appears exactly in the mesh; 6 segments x 3 = 18 elements."""
    mesh = build_mesh(ALL_TRIGGERS, 3)
    assert len(mesh) == 19
    assert set(required_nodes(ALL_TRIGGERS)) <= set(mesh)


@pytest.mark.parametrize("n", [0, 51])
def test_n_outside_1_to_50_is_rejected(n):
    """Limit: 1 <= n <= 50 elements per segment."""
    with pytest.raises(ValueError):
        build_mesh(CANTILEVER, n)


def test_n_50_is_allowed():
    """Limit: n = 50 on one segment gives 50 elements (51 nodes)."""
    assert len(build_mesh(CANTILEVER, 50)) == 51


def test_500_elements_total_is_allowed():
    """Limit: 10 segments x n = 50 = 500 elements, exactly at the cap (501 nodes)."""
    beam = replace(CANTILEVER, point_loads=loads_at([22.0 * k for k in range(1, 10)]))
    assert len(build_mesh(beam, 50)) == 501


def test_more_than_500_elements_is_rejected():
    """Limit: 11 segments x n = 46 = 506 elements, over the cap of 500."""
    beam = replace(CANTILEVER, point_loads=loads_at([20.0 * k for k in range(1, 11)]))
    with pytest.raises(ValueError):
        build_mesh(beam, 46)


def test_node_index():
    """node_index returns the node within 1e-9 mm of x."""
    mesh = build_mesh(CANTILEVER, 4)  # [0, 55, 110, 165, 220]
    assert node_index(mesh, 110.0) == 2
    assert node_index(mesh, 220.0 - 5e-10) == 4


def test_node_index_rejects_location_without_node():
    """node_index raises ValueError when no node is within 1e-9 mm of x."""
    with pytest.raises(ValueError):
        node_index(build_mesh(CANTILEVER, 4), 100.0)
