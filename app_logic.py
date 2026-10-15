"""Glue between the Streamlit page and the solver (no Streamlit here).

Converts the user's inputs (App design: point and uniform loads positive
downward, applied moments as a magnitude plus a direction) into a solver
Beam, solves it with errors captured, and prepares the summary and figure
the page shows. Units: N, mm, MPa.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from solver import (AppliedMoment, Beam, BeamResult, PointLoad, Support, UniformLoad,
                    UnstableBeamError, solve_beam)
from solver.mesh import default_n_per_segment

MOMENT_DIRECTIONS = ("counterclockwise", "clockwise")
POINTS_PER_ELEMENT = 11  # samples of the exact moment per element, ends and midpoint included


def build_beam(L, E, I, supports=(), point_loads=(), uniform_loads=(), moments=()):
    """Return the solver Beam for inputs given in the user's convention.

    Implements the App design sign rules: a point load P (N, + = downward)
    becomes F = -P; a uniform load w (N/mm, + = downward) becomes -w; an
    applied moment of magnitude |M| (N*mm) becomes +|M| if counterclockwise
    and -|M| if clockwise. Supports pass through unchanged.

    supports: (kind, x) pairs. point_loads: (x, P) pairs.
    uniform_loads: (x_start, x_end, w) triples. moments: (x, |M|, direction).
    """
    applied = []
    for x, magnitude, direction in moments:
        if direction not in MOMENT_DIRECTIONS:
            raise ValueError(
                f"Moment direction must be one of {MOMENT_DIRECTIONS}, got {direction!r}.")
        if magnitude < 0:
            raise ValueError(f"Enter the moment magnitude as a positive number, got {magnitude}.")
        applied.append(AppliedMoment(x, magnitude if direction == "counterclockwise" else -magnitude))
    return Beam(L=L, E=E, I=I,
                supports=tuple(Support(x, kind) for kind, x in supports),
                point_loads=tuple(PointLoad(x, -P) for x, P in point_loads),
                uniform_loads=tuple(UniformLoad(a, b, -w) for a, b, w in uniform_loads),
                moments=tuple(applied))


@dataclass(frozen=True)
class Outcome:
    """What the page shows after Solve: a result and the n used, or an error message."""

    result: BeamResult | None = None
    n_per_segment: int | None = None
    error: str | None = None


def solve(beam, n_per_segment=None):
    """Solve the beam, returning the result or the error message to show.

    Implements the App design error rule: a ValueError shows its own message,
    and an UnstableBeamError shows the Spec's message, with no results. With
    n_per_segment = None ("Automatic"), the Default mesh rule picks n.
    """
    try:
        result = solve_beam(beam, n_per_segment)
    except (UnstableBeamError, ValueError) as exc:
        return Outcome(error=str(exc))
    n_used = n_per_segment if n_per_segment is not None else default_n_per_segment(beam)
    return Outcome(result=result, n_per_segment=n_used)


def element_moment_terms(result):
    """Return x_left, Le, M_left, V_left and w_e for every element.

    Implements w_e = (V_right - V_left) / Le: V is linear in an element
    (dV/dx = w), so this is the element's uniform-load intensity, and it is
    exactly 0 on an element with no load.
    """
    x_left, x_right = result.x_vm[0::2], result.x_vm[1::2]
    Le = x_right - x_left
    V_left, V_right = result.V[0::2], result.V[1::2]
    return x_left, Le, result.M[0::2], V_left, (V_right - V_left) / Le


def moment_curve(result, points_per_element=POINTS_PER_ELEMENT):
    """Return x and M at several points in every element, for plotting.

    Implements the exact in-element moment M(s) = M_left + V_left*s + w_e*s^2/2
    for 0 <= s <= Le. Element ends appear twice (left and right of a node), so
    jumps at point loads and applied moments are drawn as vertical steps.
    """
    x_left, Le, M_left, V_left, w = element_moment_terms(result)
    s = np.linspace(0.0, 1.0, points_per_element)[None, :] * Le[:, None]
    M = M_left[:, None] + V_left[:, None] * s + w[:, None] * s**2 / 2
    return (x_left[:, None] + s).ravel(), M.ravel()


def moment_extremes(result):
    """Return (M_max, x) and (M_min, x), the largest and smallest moment on the beam.

    Implements the App design rule: the candidates are all element-end values,
    plus each interior peak M* = M_left - V_left^2 / (2 w_e) at
    s* = -V_left / w_e, when w_e != 0 and 0 < s* < Le (V changes sign inside
    the element). With w_e = 0, M is linear in the element: no interior peak.
    """
    xs, Ms = list(result.x_vm), list(result.M)
    for x0, Le, M0, V0, w in zip(*element_moment_terms(result)):
        if w != 0:
            s = -V0 / w
            if 0 < s < Le:
                xs.append(x0 + s)
                Ms.append(M0 - V0**2 / (2 * w))
    i_max, i_min = int(np.argmax(Ms)), int(np.argmin(Ms))
    return (float(Ms[i_max]), float(xs[i_max])), (float(Ms[i_min]), float(xs[i_min]))


@dataclass(frozen=True)
class Summary:
    """Key results for the page, each as (value, x in mm).

    v: the largest deflection by size at the nodes (mm, signed, + = up).
    M_max, M_min: the largest and smallest moment, interior peaks included
        (N*mm, sagging positive).
    V: the largest shear by size at the element ends (N, signed).
    """

    v: tuple[float, float]
    M_max: tuple[float, float]
    M_min: tuple[float, float]
    V: tuple[float, float]


def summarize(result):
    """Return the Summary of a solved beam.

    Implements: largest |v| over the nodes; M_max and M_min from
    moment_extremes; largest |V| over the element ends (V is linear inside
    an element, so its extremes are at element ends).
    """
    iv = int(np.argmax(np.abs(result.v)))
    iV = int(np.argmax(np.abs(result.V)))
    M_max, M_min = moment_extremes(result)
    return Summary(v=(float(result.v[iv]), float(result.x_nodes[iv])),
                   M_max=M_max, M_min=M_min,
                   V=(float(result.V[iV]), float(result.x_vm[iV])))


def make_figure(result):
    """Return a matplotlib Figure with deflection, shear and moment plots.

    Implements the App design drawing rules: deflection v as straight lines
    between nodes; shear V as straight lines between element ends (exact,
    since V is linear in an element); moment M from moment_curve, exact
    inside each element. Jumps appear as vertical steps.
    """
    fig = Figure(figsize=(8, 8), layout="constrained")
    ax_v, ax_V, ax_M = fig.subplots(3, 1, sharex=True)
    ax_v.plot(result.x_nodes, result.v, gid="deflection")
    ax_V.plot(result.x_vm, result.V, gid="shear")
    ax_M.plot(*moment_curve(result), gid="moment")
    for ax, label in ((ax_v, "Deflection v (mm, + = up)"),
                      (ax_V, "Shear V (N)"),
                      (ax_M, "Moment M (N*mm, sagging +)")):
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    ax_M.set_xlabel("x (mm)")
    return fig


def format_value(value, zero_tol=0.0):
    """Return value as text at 4 significant figures (App design display rule).

    Implements f"{value:.4g}"; a value with |value| <= zero_tol shows as "0",
    so floating-point noise is not displayed as a tiny number.
    """
    if abs(value) <= zero_tol:
        return "0"
    return f"{value:.4g}"
