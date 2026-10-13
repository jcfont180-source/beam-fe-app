"""Global assembly of the beam finite-element model.

Node i has two unknowns: deflection v_i at index 2i and rotation theta_i
at index 2i + 1. Element e joins nodes e and e + 1. Units: N, mm, MPa;
forces positive up, moments counterclockwise positive.
"""

import numpy as np

from solver.element import element_stiffness, uniform_load_vector
from solver.mesh import node_index


def assemble_stiffness(x_nodes, EI):
    """Return the global stiffness matrix K (2N x 2N) for nodes at x_nodes (mm).

    Implements direct stiffness assembly, K = sum over elements of
    A_e^T k_e A_e: element e joins nodes e and e + 1, so its 4x4 matrix
    k_e (from element_stiffness) is added into rows and columns 2e to 2e + 3.
    """
    K = np.zeros((2 * len(x_nodes), 2 * len(x_nodes)))
    for e, Le in enumerate(np.diff(x_nodes)):
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += element_stiffness(EI, Le)
    return K


def element_udl(beam, x_nodes):
    """Return the uniform-load intensity on each element (N/mm, positive up).

    Implements w_e = sum of w over every uniform load with
    x_start <= x_mid <= x_end, where x_mid is the midpoint of element e.
    The meshing rule puts nodes at both ends of each uniform load, so each
    element is either fully covered by a load or not covered at all.
    """
    x_nodes = np.asarray(x_nodes)
    x_mid = (x_nodes[:-1] + x_nodes[1:]) / 2
    w = np.zeros(len(x_mid))
    for u in beam.uniform_loads:
        w[(u.x_start <= x_mid) & (x_mid <= u.x_end)] += u.w
    return w


def assemble_loads(beam, x_nodes):
    """Return the global load vector F (2N) for nodes at x_nodes (mm).

    Implements F[2i] += F for a point load at node i, F[2i + 1] += M for an
    applied moment at node i, and F[2e:2e + 4] += f_eq(w_e, Le) for the
    uniform load on element e, with f_eq from uniform_load_vector.
    """
    F = np.zeros(2 * len(x_nodes))
    for p in beam.point_loads:
        F[2 * node_index(x_nodes, p.x)] += p.F
    for m in beam.moments:
        F[2 * node_index(x_nodes, m.x) + 1] += m.M
    for e, (w, Le) in enumerate(zip(element_udl(beam, x_nodes), np.diff(x_nodes))):
        F[2 * e:2 * e + 4] += uniform_load_vector(w, Le)
    return F
