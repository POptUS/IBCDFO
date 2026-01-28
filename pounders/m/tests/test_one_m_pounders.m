% This tests the ability of the MATLAB pounders to solve a one-residual
% problem. The benefit of pounders comes from its handling of the separate
% residuals, and combinging them using the known form of an outer function.
% So this is not how pounders is intended to be used, but can be a comparison
% method.

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));
addpath(fullfile(here_path, '..', 'general_h_funs'));

Ffun = @(x) ((1 - x(1)).^2) + (100 * ((x(2) - (x(1)^2)).^2)); % Rosenbrock
X_0 = [-1.2, 1];

n = 2;
np_max = 2 * n + 1;
nf_max = 200;
g_tol = 1e-8;
delta_0 = 0.1;
nfs = 1;
m = 1;
F_init = Ffun(X_0);
xk_in = 1;
Low = -Inf(1, n);
Upp = Inf(1, n);
printf = 1;
spsolver = 2;
hfun = @h_identity;
combinemodels = @combine_identity;

Prior.xk_in = xk_in;
Prior.X_0 = X_0;
Prior.F_init = F_init;
Prior.nfs = nfs;

Options.hfun = hfun;
Options.combinemodels = combinemodels;
Options.spsolver = spsolver;
Options.printf = printf;

Model.np_max = np_max;
[X, F, hf, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior, Options, Model);

path(oldpath);

assert(flag == 0, "We should solve this within 200 evaluations");
assert(norm(X(xk_best, :) - ones(1, 2)) <= g_tol * 10, "We should be within 10*gtol of the known optimum [1,1]");
