"""Checkpoint A3 tests: global stiffness and load assembly."""

import numpy as np
import pytest

from solver.model import AppliedMoment, Beam, PointLoad, Support, UniformLoad
from solver.solve import assemble_loads, assemble_stiffness, element_udl

# (EI in N*mm^2, L in mm): unit values, and the measured test bar
# (E = 68,900 MPa, I = 65.68 mm^4) with two 55 mm elements.
STIFFNESS_CASES = [
    pytest.param(1.0, 1.0, id="unit"),
    pytest.param(68900 * 65.68, 55.0, id="test-bar"),
]


def K_hand_hat(L):
    """K_hand / (EI/L^3), as written in docs/derivation/global_stiffness_assembly.m."""
    return np.array([
        [12, 6 * L, -12, 6 * L, 0, 0],
        [6 * L, 4 * L**2, -6 * L, 2 * L**2, 0, 0],
        [-12, -6 * L, 24, 0, -12, 6 * L],
        [6 * L, 2 * L**2, 0, 8 * L**2, -6 * L, 2 * L**2],
        [0, 0, -12, -6 * L, 12, -6 * L],
        [0, 0, 6 * L, 2 * L**2, -6 * L, 4 * L**2],
    ])


@pytest.mark.parametrize("EI, L", STIFFNESS_CASES)
def test_two_element_stiffness_matches_hand_assembly(EI, L):
    """K = (EI/L^3) * K_hand for two equal elements with nodes at 0, L, 2L."""
    K = assemble_stiffness(np.array([0.0, L, 2 * L]), EI)
    expected = EI / L**3 * K_hand_hat(L)
    np.testing.assert_allclose(K, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


# Load-assembly beam, meshed at its required nodes (n = 1):
#   point load      -5 N     at x = 40 mm
#   applied moment  +300 N*mm at x = 100 mm
#   uniform load A  -0.03 N/mm from x = 60 to 180 mm
#   uniform load B  -0.06 N/mm from x = 100 to 140 mm (overlaps A)
LOAD_BEAM = Beam(L=220.0, E=68900.0, I=65.68,
                 supports=[Support(0.0, "pin"), Support(220.0, "roller")],
                 point_loads=[PointLoad(40.0, -5.0)],
                 moments=[AppliedMoment(100.0, 300.0)],
                 uniform_loads=[UniformLoad(60.0, 180.0, -0.03),
                                UniformLoad(100.0, 140.0, -0.06)])
LOAD_NODES = np.array([0.0, 40.0, 60.0, 100.0, 140.0, 180.0, 220.0])


def test_element_udl():
    """w_e = sum of w over the uniform loads that cover element e's midpoint."""
    np.testing.assert_allclose(element_udl(LOAD_BEAM, LOAD_NODES),
                               [0.0, 0.0, -0.03, -0.09, -0.03, 0.0],
                               rtol=1e-12, atol=1e-15)


def test_assemble_loads():
    """F[2i] += point loads; F[2i+1] += applied moments; F[2e:2e+4] += f_eq(w_e, Le)."""
    expected = [
        0.0, 0.0,  # node 0, x = 0
        -5.0, 0.0,  # node 1, x = 40: point load
        -0.6, -4.0,  # node 2, x = 60: start of A
        -2.4, 292.0,  # node 3, x = 100: A, start of B, applied moment
        -2.4, 8.0,  # node 4, x = 140: end of B, A continues
        -0.6, 4.0,  # node 5, x = 180: end of A
        0.0, 0.0,  # node 6, x = 220
    ]
    np.testing.assert_allclose(assemble_loads(LOAD_BEAM, LOAD_NODES), expected,
                               rtol=1e-12, atol=1e-12)
