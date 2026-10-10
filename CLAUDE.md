# Beam FE solver + web app

## Spec (I write this section; don't edit it)
- Units: N, mm, MPa throughout the solver (E in MPa, I in mm^4,
  distributed loads in N/mm, moments in N·mm, rotations in radians).
- Sign conventions: x = 0 at the left end, increasing right. Deflection v
  positive up; rotation θ = dv/dx (counterclockwise positive). Inside the
  solver, forces are positive up and nodal moments counterclockwise
  positive. App users enter downward loads as positive numbers; the app
  flips the sign before calling the solver. Reported shear and moment use
  the textbook convention: sagging moment positive, shear positive when it
  tends to rotate the segment clockwise, so M = EI·v'' and V = dM/dx.
- Sanity case: cantilever fixed at x = 0, downward tip load P gives
  negative tip deflection, negative tip rotation, negative moment at the
  fixed end.
- Current phase and its gate: Setup only. Repo skeleton, no solver code.

## Working rules
- Propose a plan before writing code, and build only the current phase.
- Never change an expected value in tests/. If a test fails, fix the
  solver or stop and tell me what you found.
- NumPy only in the solver: no FEA or structural libraries. Keep the
  solver in its own module; the Streamlit app only calls it.
- Every function's docstring names the equation it implements.
- Run pytest and show me the output before calling anything done.
