% This runs goombah and goombah_wo_msp, using the LHS TRSP solver (not GAMS),
% on a tiny problem for every evaluation budget nf_max, in steps of one, from its
% smallest allowed value up through n + MAX_EXTRA. Each budget ends a run at a
% different point (while building the first models, after the first TRSP solve,
% ...), and we check that the outputs are consistent however the run ends. The
% small dimension ensures each run ends quickly.
%
% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
% and a MINQ clone and add
%    /path/to/MINQ/m/minq5
% to the MATLAB path.

function [] = test_goombah_budget_sweep()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
% Suppress the call stack that MATLAB prints with each warning.
oldbacktrace = warning('off', 'backtrace');
addpath(fullfile(repo_root, 'goombah', 'm', 'subproblems'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));
addpath(fullfile(repo_root, 'pounders', 'm'));

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

L = -5000 * ones(1, n);
U = 5000 * ones(1, n);

Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
x0 = dfoxs(n, nprob, factor^factor_power)';

hfun = @h_pw_maximum_squared;
MAX_EXTRA = 20;

for nf_max = n + 1:n + MAX_EXTRA
    problem = ['problem ' num2str(row) ' with nf_max=' int2str(nf_max)];

    rand('seed', nf_max);
    [X, F, h, ~] = goombah(hfun, Ffun, nf_max, x0, L, U, TRSP_options, subprob_switch);
    check_outputs(X, F, h, hfun, nf_max, "goombah on " + problem);

    rand('seed', nf_max);
    [X, F, h, ~] = goombah_wo_msp(hfun, Ffun, nf_max, x0, L, U, TRSP_options);
    check_outputs(X, F, h, hfun, nf_max, "goombah_wo_msp on " + problem);
end

warning(oldbacktrace);
path(oldpath);
end

function [] = check_outputs(X, F, h, hfun, nf_max, description)
    nf = size(X, 1);
    assert(nf <= nf_max, "Exceeded the evaluation budget: " + description);
    assert(size(F, 1) == nf && numel(h) == nf, "Inconsistent output sizes: " + description);
    for j = 1:nf
        assert(abs(h(j) - hfun(F(j, :))) <= 1e-12 * max(1, abs(h(j))), "h is not hfun(F) at evaluation " + int2str(j) + ": " + description);
    end
    assert(min(h) <= h(1), "Ended worse than the starting point: " + description);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
