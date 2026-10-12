"""Solver input classes and input validation.

Units: N, mm, MPa. Solver sign convention: x measured from the left end,
forces positive up, moments counterclockwise positive.
"""

import math
from dataclasses import dataclass
from typing import Sequence

from solver.mesh import MERGE_TOL, trigger_locations

SUPPORT_KINDS = ("fixed", "pin", "roller")


@dataclass(frozen=True)
class Support:
    """Support at x (mm). "fixed": v = 0 and theta = 0. "pin" or "roller": v = 0.

    Pin and roller act the same here: the solver has no axial degree of freedom.
    """

    x: float
    kind: str


@dataclass(frozen=True)
class PointLoad:
    """Point force F (N, positive up) at x (mm)."""

    x: float
    F: float


@dataclass(frozen=True)
class AppliedMoment:
    """Applied moment M (N*mm, counterclockwise positive) at x (mm)."""

    x: float
    M: float


@dataclass(frozen=True)
class UniformLoad:
    """Uniform load w (N/mm, positive up) from x_start to x_end (mm)."""

    x_start: float
    x_end: float
    w: float


@dataclass(frozen=True)
class Beam:
    """Straight beam of length L (mm) with constant E (MPa) and I (mm^4)."""

    L: float
    E: float
    I: float
    supports: Sequence[Support] = ()
    point_loads: Sequence[PointLoad] = ()
    moments: Sequence[AppliedMoment] = ()
    uniform_loads: Sequence[UniformLoad] = ()


def validate_beam(beam):
    """Check the inputs against the rules below; raise ValueError on the first one broken.

    Rules: L, E and I are positive and finite; every support, load and
    uniform-load end location x satisfies 0 <= x <= L; each uniform load
    has x_end - x_start > 1e-9 mm; each support kind is "fixed", "pin" or
    "roller"; no two supports are within 1e-9 mm of each other. An empty
    support list is allowed: the singularity check reports it as unstable.
    """
    for name in ("L", "E", "I"):
        value = getattr(beam, name)
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a positive number, got {value}.")

    for x in trigger_locations(beam):
        if not 0 <= x <= beam.L:
            raise ValueError(f"Location x = {x} mm is outside the beam (0 to {beam.L} mm).")

    for u in beam.uniform_loads:
        if u.x_end - u.x_start <= MERGE_TOL:
            raise ValueError(
                f"A uniform load must end after it starts, got {u.x_start} to {u.x_end} mm.")

    for s in beam.supports:
        if s.kind not in SUPPORT_KINDS:
            raise ValueError(f"Support kind must be one of {SUPPORT_KINDS}, got {s.kind!r}.")

    xs = sorted(s.x for s in beam.supports)
    for a, b in zip(xs, xs[1:]):
        if b - a <= MERGE_TOL:
            raise ValueError(f"Two supports at x = {a} mm; use one support per location.")
