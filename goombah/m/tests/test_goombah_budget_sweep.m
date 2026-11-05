% This runs goombah, using the LHS TRSP solver (not GAMS), on
% a tiny problem for every evaluation budget nf_max, in steps of one, from its
% smallest allowed value up through n + MAX_EXTRA. The goal is for the test to
% increase the likelihood that each execution termination path in GOOMBAH is
% exercised by the test with basic sanity checking of returned values.  Since
% the test cannot interrogate at each iteration which termination path was
% exercised, this test cannot determine how good a job it does at covering all
% paths.  Rather, users would need to manually inspect code-coverage-by-line
% reports.  The small dimension ensures each run ends quickly.
%
% To run this test, you must first install a BenDFO and a MINQ clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
%    /path/to/MINQ/m/minq5
% to the MATLAB path.

function [] = test_goombah_budget_sweep()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

factor = 10;
subprob_switch = 'linprog';
TRSP_options.solver = 'lhs';

load dfo.dat;

row = 7;
nprob = dfo(row, 1);
n = dfo(row, 2);
m = dfo(row, 3);
factor_power = dfo(row, 4);

BenDFO.nprob = nprob;
BenDFO.n = n;
BenDFO.m = m;

Low = -5000 * ones(1, n);
Upp = 5000 * ones(1, n);

Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
x0 = dfoxs(n, nprob, factor^factor_power)';

hfun = @h_pw_maximum_squared;
MAX_EXTRA = 20;

for nf_max = n + 1:n + MAX_EXTRA
    problem = ['problem ' num2str(row) ' with nf_max=' int2str(nf_max)];

    rand('seed', 1);
    [X, F, hF, ~] = goombah(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, subprob_switch);
    check_outputs(X, F, hF, hfun, nf_max, "goombah on " + problem);
end

path(oldpath);
end

function [] = check_outputs(X, F, hF, hfun, nf_max, description)
    nf = size(X, 1);
    assert(nf <= nf_max, "Exceeded the evaluation budget: " + description);
    assert(size(F, 1) == nf && numel(hF) == nf, "Inconsistent output sizes: " + description);
    for j = 1:nf
        assert(abs(hF(j) - hfun(F(j, :))) <= 1e-12 * max(1, abs(hF(j))), "h is not hfun(F) at evaluation " + int2str(j) + ": " + description);
    end
    assert(min(hF) <= hF(1), "Ended worse than the starting point: " + description);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
