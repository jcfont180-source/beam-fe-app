"""Euler-Bernoulli beam element: stiffness matrix and uniform-load vector.

Each element has two nodes and four degrees of freedom, ordered
[v1, theta1, v2, theta2]: deflection in mm (positive up) and rotation in
rad (counterclockwise positive). Units: N, mm, MPa.
"""

import numpy as np


def element_stiffness(EI, Le):
    """Return the 4x4 stiffness matrix of one beam element.

    Implements the Euler-Bernoulli element stiffness matrix derived in
    docs/derivation/beam_element_stiffness.m:

        k = (EI/Le^3) * [[ 12,    6Le,    -12,    6Le   ],
                         [ 6Le,   4Le^2,  -6Le,   2Le^2 ],
                         [-12,   -6Le,     12,   -6Le   ],
                         [ 6Le,   2Le^2,  -6Le,   4Le^2 ]]

    EI: flexural rigidity, N*mm^2. Le: element length, mm.
    """
    return EI / Le**3 * np.array([
        [12.0, 6 * Le, -12.0, 6 * Le],
        [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
        [-12.0, -6 * Le, 12.0, -6 * Le],
        [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
    ])


def uniform_load_vector(w, Le):
    """Return the equivalent nodal loads of a uniform load on one element.

    Implements the consistent (work-equivalent) load vector

        f_eq = integral from 0 to Le of N(s)^T * w ds
             = [w*Le/2, w*Le^2/12, w*Le/2, -w*Le^2/12]

    where N = [N1, N2, N3, N4] are the cubic Hermite shape functions of
    the element, with xi = s/Le:

        N1 = 1 - 3xi^2 + 2xi^3      N2 = Le*(xi - 2xi^2 + xi^3)
        N3 = 3xi^2 - 2xi^3          N4 = Le*(xi^3 - xi^2)

    w: load intensity, N/mm, positive up. Le: element length, mm.
    Returns [F1, M1, F2, M2]: forces in N (positive up) and moments in
    N*mm (counterclockwise positive).
    """
    return np.array([w * Le / 2, w * Le**2 / 12, w * Le / 2, -w * Le**2 / 12])
