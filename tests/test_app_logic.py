"""Phase C, checkpoint C1: app_logic (input conversion, solve, summary, figure).

Every expected value comes from the Spec. Loads are given in the user's
convention (App design): point and uniform loads positive downward, applied
moments as a magnitude plus a direction.
"""

import numpy as np
import pytest

from app_logic import (build_beam, format_value, make_figure, moment_curve, reaction_moment_text,
                       solve, summarize)
from solver.model import validate_beam

E, I, L = 68900.0, 65.68, 220.0  # measured test bar
REL_TOL = 1e-6
SPEC_MESSAGE = ("Beam is unstable: add supports. The current supports do not "
                "prevent rigid-body motion.")


def bar(**inputs):
    """The measured test bar with the given supports and loads, in the user's convention."""
    return build_beam(L=L, E=E, I=I, **inputs)


def approx(expected):
    """Spec relative tolerance 1e-6, purely relative."""
    return pytest.approx(expected, rel=REL_TOL, abs=0)


def plotted_line(fig, gid):
    """x and y data of the figure line with the given gid."""
    (line,) = [ln for ax in fig.axes for ln in ax.lines if ln.get_gid() == gid]
    return line.get_xdata(), line.get_ydata()


# --- Input conversion, each through a Spec case ---

def test_point_load_first_test_case():
    """P = 4.905 N downward becomes F = -4.905 N; v(L), theta(L), M(0) match the Spec."""
    beam = bar(supports=[("fixed", 0.0)], point_loads=[(220.0, 4.905)])
    assert beam.point_loads[0].F == -4.905
    outcome = solve(beam)
    assert outcome.error is None and outcome.n_per_segment == 20  # Default mesh
    r = outcome.result
    assert r.v[-1] == approx(-3.847100)
    assert r.theta[-1] == approx(-0.02623022)
    assert r.M[0] == approx(-1079.1)


def test_uniform_load_case_1():
    """w = 0.05 N/mm downward becomes w = -0.05; Case 1's v(L), theta(L), M(0), V(0)."""
    beam = bar(supports=[("fixed", 0.0)], uniform_loads=[(0.0, 220.0, 0.05)])
    assert beam.uniform_loads[0].w == -0.05
    r = solve(beam).result
    assert r.v[-1] == approx(-3.235328)
    assert r.theta[-1] == approx(-0.01960805)
    assert r.M[0] == approx(-1210.0)
    assert r.V[0] == approx(11.0)


def test_counterclockwise_moment_case_2():
    """500 N*mm counterclockwise becomes +500; Case 2's v(L), theta(L), M(0), reaction moment."""
    beam = bar(supports=[("fixed", 0.0)], moments=[(220.0, 500.0, "counterclockwise")])
    assert beam.moments[0].M == 500.0
    r = solve(beam).result
    assert r.v[-1] == approx(2.673825)
    assert r.theta[-1] == approx(0.02430750)
    assert r.M[0] == approx(500.0)
    assert r.reactions[0].moment == approx(-500.0)


def test_clockwise_moment_is_negative():
    """App design: the app sends -|M| for a clockwise moment."""
    beam = bar(supports=[("fixed", 0.0)], moments=[(220.0, 500.0, "clockwise")])
    assert beam.moments[0].M == -500.0


def test_case_3():
    """Case 3: v(110), both reactions, and the largest sagging moment 269.775 N*mm at x = 110."""
    beam = bar(supports=[("pin", 0.0), ("roller", 220.0)], point_loads=[(110.0, 4.905)])
    r = solve(beam).result
    (mid,) = np.flatnonzero(r.x_nodes == 110.0)
    assert r.v[mid] == approx(-0.2404437)
    assert [reaction.force for reaction in r.reactions] == [approx(2.4525), approx(2.4525)]
    M_max, x = summarize(r).M_max
    assert (M_max, x) == (approx(269.775), approx(110.0))


@pytest.mark.parametrize("supports", [
    pytest.param([], id="no-supports"),
    pytest.param([("pin", 0.0)], id="pin-at-0-only"),
])
def test_must_fail_beams_show_spec_message(supports):
    """Unstable beams give the Stability check message and no result."""
    outcome = solve(bar(supports=supports, point_loads=[(220.0, 4.905)]))
    assert outcome.result is None
    assert outcome.error == SPEC_MESSAGE


def test_validation_message_passes_through():
    """A load past the end of the beam gives validate_beam's own message and no result."""
    beam = bar(supports=[("fixed", 0.0)], point_loads=[(230.0, 4.905)])
    with pytest.raises(ValueError) as expected:
        validate_beam(beam)
    outcome = solve(beam)
    assert outcome.result is None
    assert outcome.error == str(expected.value)


@pytest.mark.parametrize("moment", [
    pytest.param((220.0, 500.0, "sideways"), id="unknown-direction"),
    pytest.param((220.0, -500.0, "clockwise"), id="negative-magnitude"),
])
def test_bad_moment_input_is_rejected(moment):
    """Moments need a positive magnitude and a direction of clockwise or counterclockwise."""
    with pytest.raises(ValueError):
        bar(supports=[("fixed", 0.0)], moments=[moment])


# --- Exact moment inside each element (App design) ---

def test_interior_peak_case_4_at_n_1():
    """Case 4's beam at n = 1: largest sagging moment 302.5 N*mm at x = 110 (s* = -V_left/w_e)."""
    beam = bar(supports=[("pin", 0.0), ("roller", 220.0)], uniform_loads=[(0.0, 220.0, 0.05)])
    M_max, x = summarize(solve(beam, 1).result).M_max
    assert (M_max, x) == (approx(302.5), approx(110.0))


def test_plotted_moment_at_midpoints_case_6_at_n_2():
    """Case 6's beam at n = 2: the plotted moment at each element midpoint is q x (L - x) / 2."""
    q = 0.05
    beam = bar(supports=[("pin", 0.0), ("roller", 220.0)], uniform_loads=[(0.0, 220.0, q)])
    x_plot, M_plot = plotted_line(make_figure(solve(beam, 2).result), "moment")
    for x_mid in (55.0, 165.0):
        (i,) = np.flatnonzero(np.isclose(x_plot, x_mid, rtol=0, atol=1e-9))
        assert M_plot[i] == approx(q * x_mid * (L - x_mid) / 2)


def test_no_interior_peak_when_w_is_zero_case_2():
    """Case 2: V = 0 and w_e = 0 everywhere, so M = +500 N*mm along the whole beam, no peak."""
    beam = bar(supports=[("fixed", 0.0)], moments=[(220.0, 500.0, "counterclockwise")])
    r = solve(beam, 4).result
    _, M_curve = moment_curve(r)
    np.testing.assert_allclose(M_curve, 500.0, rtol=REL_TOL)
    summary = summarize(r)
    assert summary.M_max[0] == approx(500.0)
    assert summary.M_min[0] == approx(500.0)


# --- Summary and figure ---

def test_summary_first_test_case():
    """Largest deflection v(L) = -3.847100 mm at x = 220; largest hogging M(0) = -1079.1 at x = 0."""
    beam = bar(supports=[("fixed", 0.0)], point_loads=[(220.0, 4.905)])
    summary = summarize(solve(beam).result)
    assert summary.v == (approx(-3.847100), 220.0)
    assert summary.M_min == (approx(-1079.1), 0.0)


def test_summary_case_2_deflection():
    """Case 2: the largest deflection is v(L) = +2.673825 mm (upward) at x = 220."""
    beam = bar(supports=[("fixed", 0.0)], moments=[(220.0, 500.0, "counterclockwise")])
    assert summarize(solve(beam).result).v == (approx(2.673825), 220.0)


def test_deflection_drawn_between_nodes():
    """App design: deflection is drawn as straight lines between the nodal values."""
    beam = bar(supports=[("fixed", 0.0)], point_loads=[(220.0, 4.905)])
    r = solve(beam).result
    x_plot, v_plot = plotted_line(make_figure(r), "deflection")
    np.testing.assert_array_equal(x_plot, r.x_nodes)
    np.testing.assert_array_equal(v_plot, r.v)


@pytest.mark.parametrize("value, text", [
    (-3.847100, "-3.847"),
    (2.673825, "2.674"),
    (-1079.1, "-1079"),
    (302.5, "302.5"),
])
def test_format_value_four_significant_figures(value, text):
    """App design: display at 4 significant figures."""
    assert format_value(value) == text


def test_format_value_shows_noise_as_zero():
    """A value within the zero tolerance (e.g. the Spec's 1e-9 N for shear) displays as 0."""
    assert format_value(3.6e-15, zero_tol=1e-9) == "0"


# --- Reaction-moment display text ---

def test_reaction_moment_text_counterclockwise_first_test_case():
    """First test case: the wall's reaction moment +1079.1 N*mm shows as counterclockwise."""
    beam = bar(supports=[("fixed", 0.0)], point_loads=[(220.0, 4.905)])
    (reaction,) = solve(beam).result.reactions
    assert reaction_moment_text(reaction) == f"{1079.1:.4g} (counterclockwise)"


def test_reaction_moment_text_clockwise_case_2():
    """Case 2: the wall's reaction moment -500 N*mm shows as clockwise."""
    beam = bar(supports=[("fixed", 0.0)], moments=[(220.0, 500.0, "counterclockwise")])
    (reaction,) = solve(beam).result.reactions
    assert reaction_moment_text(reaction) == f"{-500.0:.4g} (clockwise)"


def test_reaction_moment_text_pin_and_roller_case_3():
    """Case 3: a pin and a roller hold no moment, so both show "-"."""
    beam = bar(supports=[("pin", 0.0), ("roller", 220.0)], point_loads=[(110.0, 4.905)])
    reactions = solve(beam).result.reactions
    assert [reaction_moment_text(r) for r in reactions] == ["-", "-"]
