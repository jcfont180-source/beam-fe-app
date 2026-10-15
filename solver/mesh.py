"""Meshing rule from the Spec: required nodes, merging, subdivision, and the default n.

All locations are in mm, measured from the left end of the beam.
"""

import numpy as np

MERGE_TOL = 1e-9  # mm: required node locations this close are merged (rule 1)
MAX_N_PER_SEGMENT = 50  # elements per segment (rule 3)
MAX_ELEMENTS = 500  # elements in the whole beam (rule 3)
DEFAULT_N_PER_SEGMENT = 20  # elements per segment when n is not given (Default mesh)


def trigger_locations(beam):
    """Return the locations that need a node under Meshing rule 1, besides the beam ends.

    Implements rule 1's list: every support, point load and applied moment,
    and the start and end of every uniform load.
    """
    return ([s.x for s in beam.supports]
            + [p.x for p in beam.point_loads]
            + [m.x for m in beam.moments]
            + [x for u in beam.uniform_loads for x in (u.x_start, u.x_end)])


def required_nodes(beam):
    """Return the sorted required node locations after merging (mm).

    Implements Meshing rule 1: nodes at x = 0, x = L, and every trigger
    location. Going left to right, a location within MERGE_TOL = 1e-9 mm
    of the last kept node is merged into it. The beam ends stay exactly
    0 and L.
    """
    nodes = [0.0]
    for x in sorted([beam.L] + trigger_locations(beam)):
        if x - nodes[-1] > MERGE_TOL:
            nodes.append(float(x))
    nodes[-1] = float(beam.L)  # L is the largest location; a node merged with it moves to L
    return np.array(nodes)


def build_mesh(beam, n_per_segment):
    """Return every node location (mm) after subdividing the segments.

    Implements Meshing rule 3: between adjacent required nodes a and b,
    nodes at x_k = a + k*(b - a)/n for k = 0, 1, ..., n, giving n equal
    elements per segment. Limits: n is a whole number from 1 to 50, and
    n * (number of segments) <= 500; otherwise ValueError.
    """
    n = n_per_segment
    if not 1 <= n <= MAX_N_PER_SEGMENT or n != int(n):
        raise ValueError(
            f"n_per_segment must be a whole number from 1 to {MAX_N_PER_SEGMENT}, got {n}.")
    n = int(n)
    req = required_nodes(beam)
    n_elements = n * (len(req) - 1)
    if n_elements > MAX_ELEMENTS:
        raise ValueError(
            f"The mesh would have {n_elements} elements ({len(req) - 1} segments x {n}); "
            f"the maximum is {MAX_ELEMENTS}. Use a smaller n_per_segment.")
    segments = [np.linspace(a, b, n + 1)[:-1] for a, b in zip(req[:-1], req[1:])]
    return np.concatenate(segments + [req[-1:]])


def default_n_per_segment(beam):
    """Return the number of elements per segment to use when n is not given.

    Implements the Default mesh rule: n = min(20, floor(500 / number of
    segments)), so the default never exceeds the 500-element cap. Raises
    ValueError if even n = 1 would exceed the cap.
    """
    n_segments = len(required_nodes(beam)) - 1
    n = min(DEFAULT_N_PER_SEGMENT, MAX_ELEMENTS // n_segments)
    if n < 1:
        raise ValueError(
            f"The beam has {n_segments} segments; even 1 element per segment exceeds "
            f"the maximum of {MAX_ELEMENTS} elements.")
    return n


def node_index(x_nodes, x):
    """Return the index i of the node at location x (mm).

    Implements the rule-1 match |x_nodes[i] - x| <= MERGE_TOL = 1e-9 mm;
    raises ValueError if no node is that close.
    """
    i = int(np.argmin(np.abs(np.asarray(x_nodes) - x)))
    if abs(x_nodes[i] - x) > MERGE_TOL:
        raise ValueError(f"No mesh node at x = {x} mm.")
    return i
