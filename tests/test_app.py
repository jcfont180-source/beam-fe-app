"""Phase C, checkpoint C2: the Streamlit page, driven headless with AppTest.

Every expected value comes from the Spec and is compared as the page shows
it, at 4 significant figures (App design). Loads are entered in the user's
convention: point loads positive downward, moments as magnitude + direction.
"""

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app_logic import build_beam
from solver.model import validate_beam

APP = Path(__file__).resolve().parents[1] / "app.py"
SPEC_MESSAGE = ("Beam is unstable: add supports. The current supports do not "
                "prevent rigid-body motion.")


def sig4(value):
    """A Spec value at 4 significant figures, as the page shows it."""
    return f"{value:.4g}"


@pytest.fixture
def at():
    """The page, run once with its default inputs (the first test case)."""
    return AppTest.from_file(str(APP), default_timeout=60).run()


def press_solve(at):
    """Press Solve and rerun the page."""
    return at.button(key="solve").click().run()


def metric(at, label_start):
    """The one metric whose label starts with label_start."""
    (m,) = [m for m in at.metric if m.label.startswith(label_start)]
    return m


def test_default_inputs_show_first_test_case(at):
    """Defaults are the first test case: v(L) = -3.847100 mm at x = 220 and
    M(0) = -1079.1 N*mm at x = 0, with the Default mesh n = 20 per segment."""
    press_solve(at)
    assert not at.exception and not at.error
    v = metric(at, "Largest deflection")
    assert v.value == f"{sig4(-3.847100)} mm" and "x = 220 mm" in v.label
    assert "+ = up" in v.label
    hogging = metric(at, "Largest hogging moment")
    assert hogging.value == f"{sig4(-1079.1)} N·mm" and "x = 0 mm" in hogging.label
    captions = [c.value for c in at.caption]
    assert any("n = 20 elements per segment" in c for c in captions)
    assert "Rectangle: I = b*h^3/12" in captions
    assert ("The moment curve is exact between nodes; deflection is drawn as straight "
            "lines between nodes.") in captions
    assert at.get("image")  # the plots are shown


def test_case_2_shows_its_tip_deflection(at):
    """Case 2 entered on the page, 500 N*mm counterclockwise at x = 220 and no point
    load: v(L) = +2.673825 mm at x = 220."""
    at.number_input(key="n_point_loads").set_value(0)
    at.number_input(key="n_moments").set_value(1).run()
    at.number_input(key="moment_x_0").set_value(220.0)
    at.number_input(key="moment_M_0").set_value(500.0)
    at.selectbox(key="moment_dir_0").set_value("counterclockwise")
    press_solve(at)
    assert not at.exception and not at.error
    v = metric(at, "Largest deflection")
    assert v.value == f"{sig4(2.673825)} mm" and "x = 220 mm" in v.label
    assert "+ = up" in v.label


def test_pin_only_shows_stability_message_and_no_plots(at):
    """A beam with a pin only shows the Stability check message, no results, no plots."""
    at.selectbox(key="support_kind_0").set_value("pin")
    press_solve(at)
    assert not at.exception
    assert [e.value for e in at.error] == [SPEC_MESSAGE]
    assert not at.metric and not at.get("image")


def test_load_past_the_end_shows_validation_message(at):
    """A load past the end of the beam shows validate_beam's message, no crash, no plots."""
    at.number_input(key="point_x_0").set_value(230.0)
    press_solve(at)
    beam = build_beam(L=220.0, E=68900.0, I=65.68, supports=[("fixed", 0.0)],
                      point_loads=[(230.0, 4.905)])
    with pytest.raises(ValueError) as expected:
        validate_beam(beam)
    assert not at.exception
    assert [e.value for e in at.error] == [str(expected.value)]
    assert not at.metric and not at.get("image")


def test_changed_inputs_clear_stale_results(at):
    """After Solve, changing an input clears the results and shows
    "Inputs changed: press Solve"; pressing Solve again shows results."""
    press_solve(at)
    assert at.metric and at.get("image")
    at.number_input(key="point_P_0").set_value(5.0).run()
    assert not at.metric and not at.get("image")
    assert [e.value for e in at.info] == ["Inputs changed: press Solve"]
    press_solve(at)
    assert at.metric and at.get("image") and not at.info


def test_chosen_n_is_used(at):
    """Mesh "Choose n" with n = 4 solves with 4 elements per segment."""
    at.radio(key="mesh_mode").set_value("Choose n").run()
    at.number_input(key="n_per_segment").set_value(4)
    press_solve(at)
    assert any("n = 4 elements per segment" in c.value for c in at.caption)


def test_app_imports_only_streamlit_and_app_logic():
    """app.py holds no beam math: it imports only streamlit and app_logic."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert imported <= {"streamlit", "app_logic"}
