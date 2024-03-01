% This tests the ability of the MATLAB pounders to solve a one-residual
% problem. The benefit of pounders comes from its handling of the separate
% residuals, and combinging them using the known form of an outer function.
% So this is not how pounders is intended to be used, but can be a comparison
% method.

Ffun = @(x) ((1 - x(1)).^2) + (100 * ((x(2) - (x(1)^2)).^2)); % Rosenbrock
X0 = [-1.2, 1];

n = 2;
npmax = 2 * n + 1;
nfmax = 200;
gtol = 1e-8;
delta = 0.1;
nfs = 1;
m = 1;
F0 = Ffun(X0);
xkin = 1;
L = -Inf(1, n);
U = Inf(1, n);
printf = 1;
spsolver = 2;
hfun = @(F)F;
combinemodels = @identity_combine;

[X, F, hf, flag, xkin] = ...
    pounders(Ffun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);
assert(flag == 0, "We should solve this within 200 evaluations");
assert(norm(X(xkin, :) - ones(1, 2)) <= gtol * 10, "We should be within 10*gtol of the known optimum [1,1]");
