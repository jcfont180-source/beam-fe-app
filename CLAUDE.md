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
- Current phase and its gate: Phase C: web app. Gate:
  - All app_logic and app (AppTest) tests pass, using only Spec
    values.
  - Every input conversion is tested through app_logic with a Spec
    case: point load (first test case), uniform load (Case 1),
    applied moment entered as 500 N*mm counterclockwise (Case 2),
    plus Case 3 and the must-fail beams.
  - The moment between nodes is exact: Case 4's beam at n = 1 gives a
    largest sagging moment of 302.5 N*mm at x = 110, and Case 6's
    beam at n = 2 gives a plotted moment at each element midpoint
    equal to M(x) = q x (L - x)/2 (rel. tol. 1e-6).
  - Through the page: the default inputs show the first test case's
    v(L) and M(0), and Case 2's inputs show its v(L), each matching
    the Spec value to 4 significant figures; a beam with a pin only
    shows the Stability check message and no plots; a load past the
    end of the beam shows the validation message without crashing.
  - app.py contains no beam math: it only calls app_logic and the
    solver.
  - I run the app myself and can explain how each input reaches the
    solver.

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
deflection plots. The default value of n is 20 (see Default mesh).
The maximum allowed value is n = 50 per segment, and no more than
500 elements total.

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

### Stability check

- Rule (exact, for a straight beam with no hinges): the beam is
  stable if and only if it has at least one fixed support, or
  deflection is held (by any support) at two or more different
  locations. Otherwise it is unstable.
- Proof: rigid-body motion is v(x) = a + bx, theta = b. A fixed
  support at x0 gives b = 0 and then a = 0. Supports at x1 != x2 give
  b(x1 - x2) = 0, so b = 0 and a = 0. A single pin or roller gives
  only a = -b*x1, so the beam tilts freely about it.
- When: after validate_beam, before building the mesh. The check
  uses the supports only, never the stiffness matrix.
- Failure: raise UnstableBeamError with the message "Beam is
  unstable: add supports. The current supports do not prevent
  rigid-body motion." The solver never returns results for an
  unstable beam.
- App: catch UnstableBeamError and show the message to the user
  instead of a crash or plots.
- Condition number: compute cond of the reduced stiffness matrix
  after scaling it so its diagonal entries are all 1, and report it
  in the results as a diagnostic. It is never used to reject a beam.
  Phase B records the largest value found across its test cases.
- Why not a condition-number threshold: very short elements make K
  badly conditioned without making the beam unstable. A throwaway
  check with a load 0.01 mm from a support (n = 50) gave a raw
  condition number of 5.7e19, higher than truly unstable beams,
  while deflections stayed accurate to 5e-6.
- Must-fail tests:
  - No supports at all: raises UnstableBeamError.
  - Pin at x = 0 only: raises UnstableBeamError.
- Must-pass tests (solve without error):
  - Cantilever fixed at x = 0.
  - Pin at x = 0 + roller at x = L.
  - Regression: pin at x = 0 + roller at x = L, point load at
    x = 0.1 mm, n = 50 per segment.

### Phase B verification cases

All cases use the measured test bar (E = 68,900 MPa, I = 65.68 mm^4,
EI = 4,525,352 N*mm^2) and L = 220 mm unless stated. Expected values
are hand-calculated. Relative tolerance 1e-6; for expected values of
exactly 0, absolute tolerance 1e-9 N for shear and reaction forces,
and 1e-6 N*mm for moments and reaction moments. Tests use only the
expected values written here.

#### Case 1: cantilever, full-length uniform load

- Fixed at x = 0; uniform load 0.05 N/mm downward from x = 0 to
  x = 220 mm (solver input w = -0.05).
- Formulas: |v(L)| = qL^4/8EI, |theta(L)| = qL^3/6EI,
  M(x) = -q(L - x)^2/2, V(x) = +q(L - x).
- Expected at n = 1, 2, 4:
  - v(L) = -3.235328 mm
  - theta(L) = -0.01960805 rad
  - M(0) = -1210 N*mm, V(0) = +11 N
- Expected at n = 4, both the just-left and just-right values at each
  interior node:
  - x = 55: V = +8.25 N, M = -680.625 N*mm
  - x = 110: V = +5.5 N, M = -302.5 N*mm
  - x = 165: V = +2.75 N, M = -75.625 N*mm
  - x = 220 (free end, just-left value only): V = 0 N, M = 0 N*mm
- Nodal values are exact for any n (consistent uniform-load vector).
- Purpose: closes the -f_eq gap found by the A4 sabotage check.

#### Case 2: cantilever, applied tip moment

- Fixed at x = 0; applied moment M0 = 500 N*mm counterclockwise at
  x = 220 mm (solver input +500). No other loads.
- Formulas: |v(L)| = M0 L^2/2EI, |theta(L)| = M0 L/EI, M(x) = +M0,
  V(x) = 0 (pure bending: the beam bends into a smile).
- Expected at n = 1, 2, 4:
  - v(L) = +2.673825 mm
  - theta(L) = +0.02430750 rad
  - M(0) = +500 N*mm, V(0) = 0 N
  - Reaction at x = 0: force 0 N, moment -500 N*mm (solver
    convention: the wall twists clockwise; compare M(0) = +500 in
    diagram convention)
- Expected at n = 4, both the just-left and just-right values at
  x = 55, 110, 165: V = 0 N, M = +500 N*mm.
- Expected at x = 220 (free end, just-left value only): V = 0 N,
  M = +500 N*mm. M is not 0 here because the moment is applied at
  the tip.
- Zero values use the absolute tolerances above; the zero reaction
  force uses 1e-9 N.
- Nodal values are exact for any n (the exact deflection is a
  quadratic, which the cubic element contains).
- Purpose: tests the applied-moment input and its sign, and the
  opposite signs of a reaction moment and the diagram moment at the
  same point.

#### Case 3: simply supported, midspan point load

- Pin at x = 0, roller at x = 220 mm; P = 4.905 N downward at
  x = 110 mm (solver input -4.905).
- Formulas: |v(L/2)| = PL^3/48EI, |theta(0)| = |theta(L)| = PL^2/16EI,
  |M(L/2)| = PL/4, reactions P/2 each. Cross-check by symmetry: two
  cantilevers of length L/2 carrying P/2, so v(L/2) is 1/16 and
  theta(0) is 1/8 of the first test case's tip values.
- Expected at n = 1, 2, 4 (per segment):
  - v(110) = -0.2404437 mm
  - theta(0) = -0.003278778 rad, theta(220) = +0.003278778 rad
  - M(110) = +269.775 N*mm (both the just-left and just-right values)
  - M(0) = 0 N*mm, M(220) = 0 N*mm
  - V just left of x = 110: +2.4525 N; just right: -2.4525 N
    (the jump equals P)
  - V(0) = +2.4525 N; V at x = 220 (just-left value): -2.4525 N
  - Reactions: force +2.4525 N at x = 0 and at x = 220; moment 0 at
    both (pin and roller do not resist rotation)
- Zero values use the absolute tolerances above.
- Nodal values are exact for any n (the load sits at a node).
- Purpose: tests a shear jump at a point load, opposite end rotations,
  and pin and roller reactions.

#### Case 4: simply supported, full-length uniform load

- Pin at x = 0, roller at x = 220 mm; uniform load 0.05 N/mm downward
  from x = 0 to x = 220 mm (solver input w = -0.05).
- Formulas: |v(L/2)| = 5qL^4/384EI, |theta(0)| = |theta(L)| = qL^3/24EI,
  |M(L/2)| = qL^2/8, reactions qL/2 each, V(x) = qL/2 - qx.
  Cross-check against Case 1 (same q, L, EI): v(L/2) is 5/48 of
  Case 1's v(L); theta(0) and M(L/2) are 1/4 of Case 1's values.
- Expected at n = 1, 2, 4:
  - theta(0) = -0.004902013 rad, theta(220) = +0.004902013 rad
  - M(0) = 0 N*mm, M(220) = 0 N*mm
  - V(0) = +5.5 N; V at x = 220 (just-left value): -5.5 N
  - Reactions: force +5.5 N at x = 0 and at x = 220; moment 0 at both
- Expected at n = 2, 4 only (n = 1 has no node at x = 110):
  - v(110) = -0.3370134 mm
  - M(110) = +302.5 N*mm, V(110) = 0 N (both the just-left and
    just-right values)
- Zero values use the absolute tolerances above.
- Nodal values are exact for any n (consistent uniform-load vector).
- Purpose: tests a uniform load between two supports. At n = 1 every
  reported element-end value is exact, but a straight-line moment
  plot between them would be flat at 0; the app's exact in-element
  moment (App design) shows the true peak of 302.5 N*mm at x = 110.

#### Case 5: fixed at both ends, full-length uniform load

- Fixed at x = 0 and x = 220 mm; uniform load 0.05 N/mm downward from
  x = 0 to x = 220 mm (solver input w = -0.05).
- Formulas: |M(0)| = |M(L)| = qL^2/12 (hogging), M(L/2) = +qL^2/24
  (sagging), |v(L/2)| = qL^4/384EI, reactions qL/2 each. Cross-check:
  v(L/2) is 1/5 of Case 4's v(L/2).
- Expected at n = 1, 2, 4:
  - M(0) = M(220) = -201.666667 N*mm (exact value -605/3)
  - V(0) = +5.5 N; V at x = 220 (just-left value): -5.5 N
  - theta(0) = theta(220) = 0 rad exactly (held by the supports)
  - Reactions, solver convention: force +5.5 N at both walls; moment
    +201.666667 N*mm at x = 0 (counterclockwise) and -201.666667 N*mm
    at x = 220 (clockwise)
- Expected at n = 2, 4 only (n = 1 has no node at x = 110):
  - v(110) = -0.06740268 mm
  - M(110) = +100.833333 N*mm (exact value 605/6), V(110) = 0 N
    (both the just-left and just-right values)
- At n = 1 no unknown is free: the solver skips the solve, the scaled
  condition number is NaN, and every result comes from -f_eq alone
  (the fixed-end forces).
- Zero values use the absolute tolerances above.
- Purpose: tests a fixed support at the right end (its reaction
  moment sign), and the no-free-unknowns path with a load on it.

#### Case 6: convergence study (plotted moment between nodes)

- Case 4's beam: pin at x = 0, roller at x = 220 mm, uniform load
  0.05 N/mm downward over the whole span (solver input w = -0.05).
- Quantity: at the midpoint of every element, the gap between the
  exact moment M(x) = q x (L - x) / 2 (q = 0.05 N/mm, sagging
  positive) and the straight-line value between the element ends,
  (M_left + M_right) / 2, from the solver's element-end values.
- Derivation: the chord is a straight line (zero second derivative),
  so the gap has the same second derivative as M (M'' = w) and is zero
  at both element ends. That is the moment in a simply supported beam
  of length h, so the gap at every element midpoint is q h^2 / 8.
- Expected at n = 2, 4, 8, 16 (h = 220/n), at every element midpoint:
  - n = 2: 75.625 N*mm
  - n = 4: 18.90625 N*mm
  - n = 8: 4.7265625 N*mm
  - n = 16: 1.181640625 N*mm
- Each doubling of n cuts the gap by exactly 4. As a fraction of the
  true peak (302.5 N*mm) the gap is 1/n^2.
- The straight line underestimates the true sagging moment between
  nodes (unconservative).
- Relative tolerance 1e-6.

### Default mesh

- Default n = 20 elements per segment. Reason: the app draws
  deflection as straight lines between nodes and takes the largest
  deflection at nodes, so n sets how closely they follow the true
  curve. (The moment is drawn exactly inside each element; see App
  design. A straight-line moment plot would underestimate the moment
  between nodes by 1/n^2 of the peak for Case 4's beam, 0.25% at
  n = 20.) Twenty per segment still allows 25 segments under the
  500-element cap.
- solve_beam uses n = 20 when n_per_segment is not given.
- Test: solve_beam on Case 4's beam with no n argument gives the same
  results as n = 20.
- Option (b), for the default only: if n_per_segment is not given
  and 20 per segment would exceed the 500-element cap, solve_beam
  uses the largest n that fits, floor(500 / number of segments). If
  even n = 1 would exceed the cap, it raises ValueError. An n given
  explicitly is never changed; if it exceeds the cap, ValueError as
  before.
- Test: a beam with 26 segments and no n argument uses n = 19
  (494 elements).

### Phase B record

- Largest scaled condition number across all Phase B tests: 4.45e10,
  for the 26-segment default-mesh beam (n = 19, 494 elements).
- Cases 1-5 at n = 20: largest 1.26e6 (Case 3). At n = 50: largest
  4.93e7 (Case 3).
- Outside Phase B, the Phase A regression beam (load 0.1 mm from the
  pin, n = 50) gives 9.85e10, and its deflection under the load still
  matches the closed form to 1.5e-6 (relative), which supports
  reporting the condition number only, never using it to reject a
  beam.
- The condition number depends only on the supports and the mesh, not
  the loads, and grows roughly as n^4 (the beam equation
  EI v'''' = w is fourth order).
- Phase B gate passed October 15, 2026: every case passes, Case 6
  shows the 4x drop per doubling of n, the default n is set, and this
  record is complete.

### App design (Phase C)

- Loads are entered in the user's convention: point loads and uniform
  loads as positive downward (labels say "+ = downward"); the app
  sends F = -P and w = -w to the solver. Applied moments: a positive
  magnitude plus a direction dropdown (clockwise or counterclockwise);
  the app sends +|M| for counterclockwise and -|M| for clockwise.
- Section: I is entered directly (prefilled 65.68 mm^4), with the
  caption "Rectangle: I = b*h^3/12". Units: N, mm, MPa only.
- Mesh: "Automatic" (the Default mesh rule) or a chosen n from 1 to 50.
- A Solve button; results are not recomputed on every input change.
- Display: 4 significant figures; unit tests check unrounded values.
- Exact moment inside each element: V is linear in an element, so
  w_e = (V_right - V_left)/Le and M(s) = M_left + V_left*s + w_e*s^2/2
  is exact. The plot uses it at several points per element, and the
  summary's largest sagging and hogging moments include an interior
  peak at s* = -V_left/w_e when 0 < s* < Le. Deflection is drawn as
  straight lines between nodes.
- Errors: a ValueError shows its message; an UnstableBeamError shows
  the Spec's message, with no results or plots.
- app.py contains no beam math: it only calls app_logic and the
  solver. No deployment in Phase C.

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
