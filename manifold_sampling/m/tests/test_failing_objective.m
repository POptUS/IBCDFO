% This tests the ability of the MATLAB pounders implementation to
% gracefully terminate when receiving a NaN as a component in the vector of
% objective output.

function [] = test_failing_objective()

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));

nfmax = -1;
n = 3;
xs = [10; 20; 30];
L = -Inf(1, n);
U = Inf(1, n);
X0 = xs';

rand('seed', 1);
objective = @(x)failing_objective(x);
hfun = @pw_maximum_squared;

% [X, F, h, xkin, flag] = manifold_sampling_primal(hfun, objective, X0, L, U, nfmax, 'linprog');
% assert(flag == -3, "No NaN was encountered in this test, but (with high probability) should have been.");

% Intentionally not passing a function for an objective
[X, F, h, xkin, flag] = manifold_sampling_primal(hfun, X0, X0, L, U, nfmax, 'linprog');
assert(flag == -1, "Should have failed");

objective = @(x) x;
[X, F, h, xkin, flag] = manifold_sampling_primal(hfun, objective, X0, [L 1], U, nfmax, 'linprog');
assert(flag == -1, "Should have failed");

path(oldpath);
