% This tests the ability of the MATLAB pounders implementation to
% gracefully terminate when receiving a NaN as a component in the vector of
% objective output.

function [] = test_failing_objective()

spsolver = 1;

nfmax = 1000;
gtol = 1e-13;
n = 3;
m = 3;

xs = [10; 20; 30];
npmax = 2 * n + 1;
L = -Inf(1, n);
U = Inf(1, n);
nfs = 0;
X0 = xs';
F0 = [];
xkin = 1;
delta = 0.1;
printf = 0;

rand('seed', 1);
objective = @(x)failing_objective(x);

[X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver);
assert(flag == -3, "No NaN was encountered in this test, but (with high probability) should have been.");

% Intentionally not passing a function for an objective
[X, F, flag, xk_best] = pounders(X0, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver);
assert(flag == -1, "Should have failed");
