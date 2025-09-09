% This tests the ability of the MATLAB pounders implementation to
% gracefully terminate when receiving a NaN as a component in the vector of
% objective output.

function [] = test_failing_objective()

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));

spsolver = 1;

nf_max = 1000;
g_tol = 1e-13;
n = 3;
m = 3;

xs = [10; 20; 30];
np_max = 2 * n + 1;
Low = -Inf(1, n);
Upp = Inf(1, n);
X_0 = xs';
delta_0 = 0.1;

rand('seed', 1);
Ffun = @(x)failing_objective(x);

[X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp);
assert(flag == -3, "No NaN was encountered in this test, but (with high probability) should have been.");

Model.spsolver = spsolver;

% Intentionally not passing a function for an objective
[X, F, hF, flag, xk_best] = pounders(X_0, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, [], [], Model);
assert(flag == -1, "Should have failed");

% Intentionally putting a NaN in F to cover part of pounders.m
Ffun = @(x) nan(1, 3);
[X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, [], [], Model);
assert(flag == -3, "Should have failed immediately after first eval");

% Intentionally given the wrong size of F to cover part of pounders.m
Ffun = @(x) ones(1, 6);
[X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, [], [], Model);
assert(flag == -1, "Should have failed");

% Intentionally testing failures
didFail = false;
try
    [X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, [], [], 'wrong');
catch ME
    didFail = true;    
    assert(strcmp(ME.message, 'Model must be a struct'));
end
assert(didFail, 'pounders call did not fail as expected');

didFail = false;
try
    [X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, [], 'wrong');
catch ME
    didFail = true;    
    assert(strcmp(ME.message, 'Options must be a struct'));
end
assert(didFail, 'pounders call did not fail as expected');

didFail = false;
try
    [X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, 'wrong');
catch ME
    didFail = true;    
    assert(strcmp(ME.message, 'Prior must be a struct'));
end
assert(didFail, 'pounders call did not fail as expected');

path(oldpath);
