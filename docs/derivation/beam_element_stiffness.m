%% Euler-Bernoulli Beam Element Stiffness Matrix: Symbolic Derivation
% Project:      beam-fe-app (finite-element beam solver and web app)
% Author:       Juan Fontalvo
% Date:         October 2026
%
% Description:
%   Derives the 4x4 stiffness matrix of a 2-node Euler-Bernoulli beam
%   element from its cubic deflection shape, then verifies the result
%   with five physical checks.
%
% Degrees of freedom (per element):
%   d = [v1; theta1; v2; theta2]
%   v      transverse deflection [mm], positive up
%   theta  rotation dv/dx [rad], counterclockwise positive
%
% Method:
%   1. With no load inside the element, EI*v'''' = 0, so
%        v(x) = a0 + a1*x + a2*x^2 + a3*x^3,   0 <= x <= L
%   2. Matching v and v' at both nodes gives a0 = v1, a1 = theta1,
%      and a 2x2 system for a2 and a3 (solved by hand).
%   3. Internal actions, textbook convention (sagging M positive,
%      clockwise-rotating V positive):  M = EI*v'',  V = EI*v'''
%   4. End forces, solver convention (up and counterclockwise positive):
%        F1 = +V,   M1 = -M(0),   F2 = -V,   M2 = +M(L)
%   5. k is the coefficient matrix of [F1; M1; F2; M2] with respect to d.
%
% Units:    N, mm, MPa  (EI in N*mm^2)
% Requires: Symbolic Math Toolbox, MATLAB R2016b or newer

clc; clear; close all;

%% 1. Symbolic variables
syms EI L v1 theta1 v2 theta2

%% 2. Cubic coefficients from the nodal conditions (derived by hand)
a2 =  3*(v2 - v1)/L^2 - (2*theta1 + theta2)/L;
a3 = -2*(v2 - v1)/L^3 + (theta1 + theta2)/L^2;

%% 3. Internal moment and shear (textbook convention)
M_0 = 2*EI*a2;               % M(0) = EI*v''(0)
M_L = EI*(2*a2 + 6*a3*L);    % M(L) = EI*v''(L)
V   = 6*EI*a3;               % V = EI*v''', constant along the element

%% 4. End forces (solver convention: up and counterclockwise positive)
F1 =  V;       % positive V pushes up on the left face
M1 = -M_0;     % positive M twists clockwise on the left face
F2 = -V;       % positive V pushes down on the right face
M2 =  M_L;     % positive M twists counterclockwise on the right face

%% 5. Element stiffness matrix
forces = [F1; M1; F2; M2];
d      = [v1; theta1; v2; theta2];
k      = equationsToMatrix(forces, d);
k_hat  = simplify(k / (EI/L^3));    % k = (EI/L^3) * k_hat

fprintf('Element stiffness matrix:  k = (EI/L^3) * k_hat\n\n');
fprintf('k_hat =\n');
disp(k_hat);

%% 6. Verification checks
% Each residual below must simplify to all zeros.
checks = {
    'Symmetry: k - k^T',                                       k - k.';
    'Vertical equilibrium: row 1 + row 3',                     k(1,:) + k(3,:);
    'Rigid translation: k*[1; 0; 1; 0]',                       k*[1; 0; 1; 0];
    'Rigid rotation: k*[0; 1; L; 1]',                          k*[0; 1; L; 1];
    'Moment equilibrium about node 1: row 2 + row 4 + L*row 3', k(2,:) + k(4,:) + L*k(3,:)
    };

fprintf('\nVerification checks\n');
fprintf('%s\n', repmat('-', 1, 64));

allPassed = true;
for i = 1:size(checks, 1)
    passed = isZero(checks{i, 2});
    allPassed = allPassed && passed;
    if passed
        status = 'PASS';
    else
        status = 'FAIL';
    end
    fprintf('%-58s %s\n', checks{i, 1}, status);
    if ~passed
        fprintf('  Residual (should be zero):\n');
        disp(simplify(checks{i, 2}));
    end
end

fprintf('%s\n', repmat('-', 1, 64));
if allPassed
    fprintf('All checks passed.\n');
else
    fprintf('At least one check FAILED. Review the derivation.\n');
end

%% Local functions
function passed = isZero(expr)
% isZero  True if a symbolic expression simplifies to all zeros.
passed = all(isAlways(simplify(expr(:)) == 0));
end