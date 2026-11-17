% This tests the exit flag returned by manifold_sampling_primal:
%   0 or -6 if the run stops before exhausting its budget,
%   1 if the budget is exhausted, and
%   -1 if the objective fails.
%
% The converging case is row 1 of the More and Wild SIOPT benchmark
% "Benchmarking derivative-free optimization algorithms", which is linear in x
% and has the known minimum m - n = 36.
%
% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
% to the MATLAB path.

function [] = test_manifold_sampling_flags()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));
addpath(fullfile(here_path, 'general_smooth_h_funs'));

subprob_switch = 'linprog';

load dfo.dat;

row = 1;
nprob = dfo(row, 1);
n = dfo(row, 2);
m = dfo(row, 3);
factor_power = dfo(row, 4);
factor = 10;

BenDFO.nprob = nprob;
BenDFO.n = n;
BenDFO.m = m;

Low = -5000 * ones(1, n);
Upp = 5000 * ones(1, n);
hfun = @h_leastsquares;
Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
x0 = dfoxs(n, nprob, factor^factor_power)';
known_min = m - n;

% Failing objective
bad_Ffun = @(x)failing_Ffun(x);
oldwarn = warning('off', 'all'); % The solver warns that Ffun failed
[~, ~, ~, ~, flag] = manifold_sampling_primal(hfun, bad_Ffun, x0, Low, Upp, 300, subprob_switch);
warning(oldwarn);
assert(flag == -1, "flag should be -1 when Ffun fails");

% Exhausted budget: flag of 1, and outputs have nf rows
nf_max = n + 5;
[X, F, hF, ~, flag] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch);
assert(flag == 1, "flag should be 1 when the budget is exhausted");
assert(size(X, 1) == nf_max && size(F, 1) == nf_max && numel(hF) == nf_max, "outputs should have nf rows");

% Converges before exhausting the budget
nf_max = 300;
out = evalc('[X, F, hF, xk_in, flag] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch);');
assert(size(X, 1) < nf_max, "manifold_sampling_primal used its entire evaluation budget");
assert(size(F, 1) == size(X, 1) && numel(hF) == size(X, 1), "outputs should have nf rows");
assert(abs(hF(xk_in) - known_min) <= 1e-6, "manifold_sampling_primal did not converge to the known minimum");
% Normally the run stops when the radius reaches delta_min (-6 unless the
% stationary measure also satisfies g_tol, in which case the flag is 0)
assert(flag == 0 || flag == -6, "flag should be 0 or -6 when the budget is not exhausted");
if flag == -6
    assert(contains(out, 'Final chi_k = '), "The -6 message should report the final chi_k");
end

path(oldpath);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end

function [fvec] = failing_Ffun(x)
    error('Intentional failure of Ffun');
    fvec = x;
end
