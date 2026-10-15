"""Assembly, solution, and results of the beam finite-element model.

Node i has two unknowns: deflection v_i at index 2i and rotation theta_i
at index 2i + 1. Element e joins nodes e and e + 1. Units: N, mm, MPa;
forces positive up, moments counterclockwise positive.
"""

import numpy as np

from solver.element import element_stiffness, uniform_load_vector
from solver.mesh import build_mesh, default_n_per_segment, node_index
from solver.model import BeamResult, Reaction, check_stability, validate_beam


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


def constrained_dofs(beam, x_nodes):
    """Return the sorted indices of the unknowns held by the supports.

    Implements the support conditions: a fixed support at node i holds
    v_i = 0 and theta_i = 0 (indices 2i and 2i + 1); a pin or roller holds
    v_i = 0 only (index 2i).
    """
    held = []
    for s in beam.supports:
        i = node_index(x_nodes, s.x)
        held += [2 * i, 2 * i + 1] if s.kind == "fixed" else [2 * i]
    return sorted(held)


def scaled_condition_number(K_ff):
    """Return the condition number of K_ff after scaling its diagonal to 1.

    Implements cond_2(D^-1/2 K_ff D^-1/2) = sigma_max / sigma_min, with
    D = diag(K_ff). Diagnostic only, never used to reject a beam. Returns
    NaN if K_ff is empty (no unknown is free to move).
    """
    if K_ff.size == 0:
        return float("nan")
    d = 1.0 / np.sqrt(np.diag(K_ff))
    return float(np.linalg.cond(d[:, None] * K_ff * d[None, :]))


def support_reactions(beam, x_nodes, K, u, F):
    """Return one Reaction per support, in solver convention.

    Implements R = K u - F, the forces and moments the supports apply to
    the beam: force R[2i] at every support, moment R[2i + 1] at a fixed
    support (0 at a pin or roller, which does not hold rotation).
    """
    R = K @ u - F
    reactions = []
    for s in beam.supports:
        i = node_index(x_nodes, s.x)
        moment = R[2 * i + 1] if s.kind == "fixed" else 0.0
        reactions.append(Reaction(x=s.x, kind=s.kind, force=float(R[2 * i]),
                                  moment=float(moment)))
    return tuple(reactions)


def internal_forces(x_nodes, EI, w, u):
    """Return x_vm, V, M at both ends of every element, textbook convention.

    Implements the element end forces [F1, M1, F2, M2] = k_e d_e - f_eq(w_e, Le)
    (solver convention), then inverts the mapping derived in
    docs/derivation/beam_element_stiffness.m: V_left = F1, M_left = -M1,
    V_right = -F2, M_right = M2.
    """
    n_elements = len(x_nodes) - 1
    x_vm, V, M = np.empty(2 * n_elements), np.empty(2 * n_elements), np.empty(2 * n_elements)
    for e, Le in enumerate(np.diff(x_nodes)):
        d_e = u[2 * e:2 * e + 4]
        F1, M1, F2, M2 = element_stiffness(EI, Le) @ d_e - uniform_load_vector(w[e], Le)
        x_vm[2 * e:2 * e + 2] = x_nodes[e], x_nodes[e + 1]
        V[2 * e:2 * e + 2] = F1, -F2
        M[2 * e:2 * e + 2] = -M1, M2
    return x_vm, V, M


def solve_beam(beam, n_per_segment=None):
    """Solve the beam and return a BeamResult.

    Implements K_ff u_f = F_f, the global system with the rows and columns
    held by the supports removed (held unknowns are 0). Steps: validate the
    inputs, check stability from the supports, build the mesh, assemble K
    and F, solve, then compute reactions and shear and moment.

    n_per_segment: elements per segment. If not given, the Default mesh rule
    applies (default_n_per_segment): 20, or the largest n that fits the
    500-element cap. An n given explicitly is never changed.
    """
    validate_beam(beam)
    check_stability(beam)
    if n_per_segment is None:
        n_per_segment = default_n_per_segment(beam)
    x_nodes = build_mesh(beam, n_per_segment)
    EI = beam.E * beam.I
    K = assemble_stiffness(x_nodes, EI)
    F = assemble_loads(beam, x_nodes)
    free = np.setdiff1d(np.arange(len(F)), constrained_dofs(beam, x_nodes))
    K_ff = K[np.ix_(free, free)]
    u = np.zeros(len(F))
    if len(free) > 0:
        u[free] = np.linalg.solve(K_ff, F[free])
    x_vm, V, M = internal_forces(x_nodes, EI, element_udl(beam, x_nodes), u)
    return BeamResult(x_nodes=x_nodes, v=u[0::2], theta=u[1::2],
                      reactions=support_reactions(beam, x_nodes, K, u, F),
                      x_vm=x_vm, V=V, M=M, cond=scaled_condition_number(K_ff))
