"""Beam FE web app: the Streamlit page (inputs, Solve, results, plots).

This page holds no beam math. It reads the inputs in the user's convention,
calls app_logic (which calls the solver), and shows what comes back.
Run it with:  streamlit run app.py
"""

import streamlit as st

import app_logic

ZERO_TOL_V = app_logic.ZERO_TOL_V  # N: Spec zero tolerance for shear and reaction forces
ZERO_TOL_M = app_logic.ZERO_TOL_M  # N*mm: Spec zero tolerance for moments
MAX_ROWS = 20  # most supports, point loads, uniform loads or moments per list
SUPPORT_KINDS = ["fixed", "pin", "roller"]
STALE_MESSAGE = "Inputs changed: press Solve"
fmt = app_logic.format_value


def read_inputs():
    """Draw the input widgets in the sidebar and return their values.

    Implements the App design input rules: loads in the user's convention
    (point and uniform loads + = downward; moments as a positive magnitude and
    a direction), I entered directly, and the mesh "Automatic" or a chosen n.
    """
    sb = st.sidebar
    sb.header("Beam")
    L = sb.number_input("Length L (mm)", value=220.0, format="%g", key="L")
    E = sb.number_input("E (MPa)", value=68900.0, format="%g", key="E")
    I = sb.number_input("I (mm^4)", value=65.68, format="%g", key="I")
    sb.caption("Rectangle: I = b*h^3/12")

    sb.header("Supports")
    supports = []
    for i in range(sb.number_input("Number of supports", 0, MAX_ROWS, 1, key="n_supports")):
        left, right = sb.columns(2)
        kind = left.selectbox(f"Support {i + 1}", SUPPORT_KINDS, index=0 if i == 0 else 2,
                              key=f"support_kind_{i}")
        x = right.number_input("x (mm)", value=0.0 if i == 0 else L, format="%g",
                               key=f"support_x_{i}")
        supports.append((kind, x))

    sb.header("Point loads")
    point_loads = []
    for i in range(sb.number_input("Number of point loads", 0, MAX_ROWS, 1, key="n_point_loads")):
        left, right = sb.columns(2)
        x = left.number_input(f"Load {i + 1}: x (mm)", value=L if i == 0 else 0.0, format="%g",
                              key=f"point_x_{i}")
        P = right.number_input("P (N, + = downward)", value=4.905 if i == 0 else 0.0, format="%g",
                               key=f"point_P_{i}")
        point_loads.append((x, P))

    sb.header("Uniform loads")
    uniform_loads = []
    for i in range(sb.number_input("Number of uniform loads", 0, MAX_ROWS, 0, key="n_uniform_loads")):
        start, end, intensity = sb.columns(3)
        a = start.number_input(f"Load {i + 1}: from x (mm)", value=0.0, format="%g",
                               key=f"udl_start_{i}")
        b = end.number_input("to x (mm)", value=L, format="%g", key=f"udl_end_{i}")
        w = intensity.number_input("w (N/mm, + = downward)", value=0.0, format="%g",
                                   key=f"udl_w_{i}")
        uniform_loads.append((a, b, w))

    sb.header("Applied moments")
    moments = []
    for i in range(sb.number_input("Number of moments", 0, MAX_ROWS, 0, key="n_moments")):
        at_x, size, turn = sb.columns(3)
        x = at_x.number_input(f"Moment {i + 1}: x (mm)", value=L, format="%g", key=f"moment_x_{i}")
        M = size.number_input("|M| (N*mm)", min_value=0.0, value=0.0, format="%g",
                              key=f"moment_M_{i}")
        direction = turn.selectbox("Direction", app_logic.MOMENT_DIRECTIONS,
                                   format_func=str.capitalize, key=f"moment_dir_{i}")
        moments.append((x, M, direction))

    sb.header("Mesh")
    mode = sb.radio("Elements per segment", ["Automatic", "Choose n"], key="mesh_mode",
                    help="Automatic: 20 per segment, fewer if needed to stay within 500 elements.")
    n = None
    if mode == "Choose n":
        n = sb.number_input("n (1 to 50)", 1, 50, 20, key="n_per_segment")

    return {"L": L, "E": E, "I": I, "supports": supports, "point_loads": point_loads,
            "uniform_loads": uniform_loads, "moments": moments, "n_per_segment": n}


def solve_inputs(inputs):
    """Build the solver Beam from the inputs and solve it, both through app_logic.

    Implements the App design error rule: any ValueError becomes a message to
    show instead of results.
    """
    try:
        beam = app_logic.build_beam(inputs["L"], inputs["E"], inputs["I"], inputs["supports"],
                                    inputs["point_loads"], inputs["uniform_loads"],
                                    inputs["moments"])
    except ValueError as exc:
        return app_logic.Outcome(error=str(exc))
    return app_logic.solve(beam, inputs["n_per_segment"])


def show_outcome(outcome):
    """Show the error message, or the summary, reactions, plots and mesh notes.

    Implements the App design display rules: 4 significant figures, values
    within the Spec zero tolerances shown as 0, and no results or plots when
    there is an error.
    """
    if outcome.error is not None:
        st.error(outcome.error)
        return
    result = outcome.result
    summary = app_logic.summarize(result)
    (v, x_v), (M_max, x_max), (M_min, x_min), (V, x_V) = (
        summary.v, summary.M_max, summary.M_min, summary.V)

    tiles = st.columns(4)
    tiles[0].metric(f"Largest deflection (+ = up, x = {fmt(x_v)} mm)", f"{fmt(v)} mm")
    if M_max > ZERO_TOL_M:
        tiles[1].metric(f"Largest sagging moment (x = {fmt(x_max)} mm)", f"{fmt(M_max)} N·mm")
    else:
        tiles[1].metric("Largest sagging moment", "none")
    if M_min < -ZERO_TOL_M:
        tiles[2].metric(f"Largest hogging moment (x = {fmt(x_min)} mm)", f"{fmt(M_min)} N·mm")
    else:
        tiles[2].metric("Largest hogging moment", "none")
    tiles[3].metric(f"Largest shear (x = {fmt(x_V)} mm)", f"{fmt(V, ZERO_TOL_V)} N")

    st.subheader("Reactions")
    st.table([{"x (mm)": fmt(r.x), "Support": r.kind,
               "Force (N, + = up)": fmt(r.force, ZERO_TOL_V),
               "Moment (N·mm)": app_logic.reaction_moment_text(r)} for r in result.reactions])

    st.pyplot(app_logic.make_figure(result))
    st.caption("The moment curve is exact between nodes; deflection is drawn as straight "
               "lines between nodes.")
    # NaN (nothing can move, so there is no reduced matrix) is the only value not equal to itself.
    cond = fmt(result.cond) if result.cond == result.cond else "not applicable (nothing can move)"
    st.caption(f"Mesh: n = {outcome.n_per_segment} elements per segment, "
               f"{len(result.x_nodes) - 1} elements. Scaled condition number (diagnostic): {cond}.")


st.set_page_config(page_title="Beam FE solver", layout="wide")
st.title("Beam FE solver")
st.caption("Units: N, mm, MPa. Point and uniform loads: + = downward. "
           "Euler-Bernoulli beam elements, solved with NumPy.")

inputs = read_inputs()
if st.sidebar.button("Solve", type="primary", key="solve"):
    st.session_state.outcome = solve_inputs(inputs)
    st.session_state.solved_inputs = inputs
    st.session_state.stale = False
elif st.session_state.get("solved_inputs") is not None and st.session_state.solved_inputs != inputs:
    # App design: results for other inputs are cleared, never shown as current.
    del st.session_state["outcome"], st.session_state["solved_inputs"]
    st.session_state.stale = True

if "outcome" in st.session_state:
    show_outcome(st.session_state.outcome)
elif st.session_state.get("stale"):
    st.info(STALE_MESSAGE)
else:
    st.info("Enter a beam in the sidebar and press Solve.")
