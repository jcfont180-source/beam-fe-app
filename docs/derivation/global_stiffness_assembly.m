%% ========================================================================
%  Title:       Global Stiffness Assembly - Two-Element Beam
%  Author:      Juan Fontalvo
%  Date:        October 12, 2026
%  Description: Assembles the 6x6 global stiffness matrix K from two equal
%               Euler-Bernoulli beam elements, compares it with the hand
%               assembly, and verifies symmetry, both rigid-body modes,
%               rank, and determinant.
%  DOFs:        [v1 th1 v2 th2 v3 th3], nodes at x = 0, L, 2L
%  Method:      Direct stiffness assembly (element k added into global K)
%  Units:       Symbolic; numeric section uses N, mm, MPa
%  Requires:    Symbolic Math Toolbox
%% ========================================================================
clc; clear; close all;

%% 1. Element stiffness matrix
syms E I L real positive
c = E*I/L^3;
k = c*[ 12    6*L   -12    6*L;
        6*L   4*L^2 -6*L   2*L^2;
       -12   -6*L    12   -6*L;
        6*L   2*L^2 -6*L   4*L^2];

%% 2. Assembly (ADD into the global matrix, never replace)
K = sym(zeros(6));
dofs1 = 1:4;    % element 1: node 1 -> node 2
dofs2 = 3:6;    % element 2: node 2 -> node 3
K(dofs1,dofs1) = K(dofs1,dofs1) + k;
K(dofs2,dofs2) = K(dofs2,dofs2) + k;

fprintf('Assembled K = (EI/L^3) *\n');
disp(simplify(K/c));

%% 3. Hand assembly (from my notes)
K_hand = c*[ 12    6*L   -12    6*L     0     0;
             6*L   4*L^2 -6*L   2*L^2   0     0;
            -12   -6*L    24    0     -12     6*L;
             6*L   2*L^2  0     8*L^2  -6*L   2*L^2;
             0     0     -12   -6*L    12    -6*L;
             0     0      6*L   2*L^2  -6*L   4*L^2];

%% 4. Predictions
u_trans = [1; 0; 1; 0; 1; 0];        % rigid vertical translation
u_rot   = [0; 1; L; 1; 2*L; 1];      % rigid rotation about node 1, slope 1
rank_predicted = 4;                  % 6 DOFs - 2 rigid-body modes

%% 5. Checks
isZero = @(expr) all(isAlways(simplify(expr(:)) == 0));

r = double(rank(K));
d = simplify(det(K));

names = {'Matches hand assembly'
         'Symmetric (K = K'')'
         'Rigid translation: K*u_trans = 0'
         'Rigid rotation:    K*u_rot   = 0'
         sprintf('Rank = %d (predicted %d)', r, rank_predicted)
         'det(K) = 0'};
pass = [isZero(K - K_hand)
        isZero(K - K.')
        isZero(K*u_trans)
        isZero(K*u_rot)
        r == rank_predicted
        isZero(d)];

fprintf('\n%-40s %s\n', 'Check', 'Result');
fprintf('%s\n', repmat('-', 1, 48));
for i = 1:numel(names)
    if pass(i), s = 'PASS'; else, s = 'FAIL'; end
    fprintf('%-40s %s\n', names{i}, s);
end

if ~pass(1)
    fprintf('\nDifference K - K_hand (EI/L^3 factored out):\n');
    disp(simplify((K - K_hand)/c));
end

%% 6. Numeric version: why det == 0 is a bad test in floating point
%  Measured test bar: E = 68,900 MPa, I = 65.68 mm^4, element L = 100 mm
Kn = double(subs(K, [E I L], [68900 65.68 100]));
fprintf('\nNumeric K (E = 68900 MPa, I = 65.68 mm^4, L = 100 mm):\n');
fprintf('  det(Kn)  = %g\n', det(Kn));
fprintf('  rank(Kn) = %d\n', rank(Kn));
fprintf('  cond(Kn) = %g\n', cond(Kn));
