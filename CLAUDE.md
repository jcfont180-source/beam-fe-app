# Beam FE solver + web app

## Spec (I write this section; don't edit it)
- Units:
- Sign conventions (deflection, load, moment):
- Current phase and its gate: Setup only. Repo skeleton, no solver code.

## Working rules
- Propose a plan before writing code, and build only the current phase.
- Never change an expected value in tests/. If a test fails, fix the
  solver or stop and tell me what you found.
- NumPy only in the solver: no FEA or structural libraries. Keep the
  solver in its own module; the Streamlit app only calls it.
- Every function's docstring names the equation it implements.
- Run pytest and show me the output before calling anything done.
