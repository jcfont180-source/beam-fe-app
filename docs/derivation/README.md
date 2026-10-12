# Beam element stiffness derivation

`beam_element_stiffness.m` derives the 4x4 stiffness matrix of a 2-node
Euler-Bernoulli beam element from a cubic deflection shape whose
coefficients were derived by hand.

Requires MATLAB with the Symbolic Math Toolbox.

The script runs five checks on the result, all of which pass:

1. Symmetry
2. Vertical equilibrium
3. Rigid translation
4. Rigid rotation
5. Moment equilibrium

## Global stiffness assembly

`global_stiffness_assembly.m` assembles the 6x6 global stiffness matrix
of a beam made of two equal elements. It runs six checks, all of which
pass:

1. Matches hand assembly
2. Symmetry
3. Rigid translation
4. Rigid rotation
5. Rank = 4
6. det = 0

In floating point, det can print as exactly 0 by coincidence, so
singularity of a numeric stiffness matrix should be tested with the
condition number instead.
