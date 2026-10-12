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

### Meshing rule

#### 1. Required node locations (v1)
To ensure structural correctness and mathematical alignment with the
solver, nodes must be placed at:
- Supports: to allow the solver to enforce boundary conditions by
  striking out the appropriate rows and columns.
- Point loads and applied moments: to allow external forces to drop
  directly into the global nodal force vector without missing peak
  moments.
- Distributed load boundaries: at the exact start and end coordinates
  of any uniform load, so equivalent nodal forces are not applied over
  empty spans.
- Beam ends: at the left and right ends of the physical beam.

Required node locations that coincide (within 1e-9 mm) are merged
into one node.

#### 2. Out of scope for v1
The following node triggers are acknowledged but explicitly excluded
from version 1:
- Changes in cross-section (I): v1 assumes a single, continuous
  rectangular profile.
- Changes in material (E): v1 assumes a homogeneous material
  throughout the entire beam.
- Internal hinges: v1 does not support hinges, as they require an
  architectural change to the matrix assembly to track an uncoupled,
  extra rotational DOF.

#### 3. Subdivision rule
Between any two adjacent required nodes, the segment is subdivided
into n equal-length elements (n elements per segment, set by user
input or a default value). While the required nodes place every load
and support exactly, subdividing the space between them provides the
resolution needed to render smooth, continuous shear, moment, and
deflection plots. The default value of n will be set by the Phase B
convergence study.

### First test case (Decision 4)

Cantilever fixed at x = 0, point load at x = L, measured test bar.
- E = 68,900 MPa, I = 65.68 mm^4 (measured; see
  docs/measurements/bar_dimensions.md), L = 220 mm
- Load: 500 g, P = 0.500 x 9.81 = 4.905 N downward
  (solver input: -4.905 N)
- Expected (hand-calculated, standard cantilever formulas):
  - Tip deflection v(L) = -3.847100 mm   (|v| = PL^3/3EI)
  - Tip rotation theta(L) = -0.02623022 rad   (|theta| = PL^2/2EI)
  - Fixed-end moment M(0) = -1079.1 N*mm   (|M| = PL, hogging)
- Run with 1, 2, and 4 elements; all must pass. FE is exact for this
  case (the exact deflection is cubic, matching the element shape
  functions), so any error beyond floating-point noise is a bug.
- Relative tolerance: 1e-6.

## Working rules
- Propose a plan before writing code, and build only the current phase.
- Never change an expected value in tests/. If a test fails, fix the
  solver or stop and tell me what you found.
- NumPy only in the solver: no FEA or structural libraries. Keep the
  solver in its own module; the Streamlit app only calls it.
- Every function's docstring names the equation it implements.
- Run pytest and show me the output before calling anything done.
- If I ask to see a diff before committing, stop after showing it and
  wait for my explicit approval. Do not commit or push in the same turn.
