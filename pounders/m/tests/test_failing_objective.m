% This tests the ability of the MATLAB pounders implementation to
% gracefully terminate when receiving a NaN as a component in the vector of
% objective output.

function [] = test_failing_objective()

spsolver = 1;

nf_max = 1000;
g_tol = 1e-13;
n = 3;
m = 3;

xs = [10; 20; 30];
np_max = 2 * n + 1;
Low = -Inf(1, n);
Upp = Inf(1, n);
nfs = 0;
X0 = xs';
F0 = [];
xk_in = 1;
delta = 0.1;
printf = 0;

rand('seed', 1);
objective = @(x)failing_objective(x);

[X, F, flag, xk_best] = pounders(objective, X0, n, np_max, nf_max, g_tol, delta, nfs, m, F0, xk_in, Low, Upp, printf, spsolver);
assert(flag == -3, "No NaN was encountered in this test, but (with high probability) should have been.");

% Intentionally not passing a function for an objective
[X, F, flag, xk_best] = pounders(X0, X0, n, np_max, nf_max, g_tol, delta, nfs, m, F0, xk_in, Low, Upp, printf, spsolver);
assert(flag == -1, "Should have failed");
