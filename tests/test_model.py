"""Checkpoint A2 tests: input validation."""

from dataclasses import replace

import pytest

from solver.model import AppliedMoment, Beam, PointLoad, Support, UniformLoad, validate_beam

# First test case: cantilever fixed at x = 0, 4.905 N downward at x = L.
CANTILEVER = Beam(L=220.0, E=68900.0, I=65.68,
                  supports=[Support(0.0, "fixed")],
                  point_loads=[PointLoad(220.0, -4.905)])


def test_valid_beam_passes():
    """A beam that meets every input rule raises nothing."""
    validate_beam(CANTILEVER)


def test_no_supports_is_not_an_input_error():
    """Empty supports pass validation; the singularity check reports them (decision 3)."""
    validate_beam(replace(CANTILEVER, supports=[]))


@pytest.mark.parametrize("beam", [
    pytest.param(replace(CANTILEVER, L=0.0), id="L-zero"),
    pytest.param(replace(CANTILEVER, E=-68900.0), id="E-negative"),
    pytest.param(replace(CANTILEVER, I=0.0), id="I-zero"),
    pytest.param(replace(CANTILEVER, supports=[Support(-1.0, "fixed")]),
                 id="support-before-start"),
    pytest.param(replace(CANTILEVER, point_loads=[PointLoad(230.0, -4.905)]),
                 id="load-past-end"),
    pytest.param(replace(CANTILEVER, moments=[AppliedMoment(221.0, 100.0)]),
                 id="moment-past-end"),
    pytest.param(replace(CANTILEVER, uniform_loads=[UniformLoad(0.0, 230.0, -0.01)]),
                 id="uniform-load-past-end"),
    pytest.param(replace(CANTILEVER, uniform_loads=[UniformLoad(150.0, 100.0, -0.01)]),
                 id="uniform-load-reversed"),
    pytest.param(replace(CANTILEVER, uniform_loads=[UniformLoad(100.0, 100.0, -0.01)]),
                 id="uniform-load-zero-length"),
    pytest.param(replace(CANTILEVER, supports=[Support(0.0, "hinge")]),
                 id="unknown-support-kind"),
    pytest.param(replace(CANTILEVER, supports=[Support(0.0, "fixed"), Support(0.0, "pin")]),
                 id="two-supports-same-x"),
])
def test_invalid_beam_raises_value_error(beam):
    """Each broken input rule raises ValueError."""
    with pytest.raises(ValueError):
        validate_beam(beam)
