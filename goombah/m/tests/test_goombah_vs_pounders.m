% This compares goombah and goombah_wo_msp, using the LHS TRSP solver instead of
% GAMS, against pounders on least-squares problems h(F(x)) = sum(F(x).^2) from
% the More and Wild SIOPT paper "Benchmarking derivative-free optimization
% algorithms". Least squares is the only h function tested to which pounders applies.
%
% For each problem, we assert that each GOOMBAH variant decreases h at least 90%
% as much as pounders does from the same starting point.
%
% To run this test, you must first install a BenDFO and a MINQ clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
%    /path/to/MINQ/m/minq5
% to the MATLAB path.

function [] = test_goombah_vs_pounders()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'pounders', 'm'));
% While both POUNDERS and Manifold Sampling provide an h_leastsquares, both of
% which are used here, the two functions have different interfaces.
addpath(fullfile(repo_root, 'pounders', 'm', 'general_h_funs'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'tests', 'general_smooth_h_funs'));

factor = 10;
nf_max = 100;
g_tol = 1e-13;
delta_0 = 0.1;
spsolver = 2; % TRSP Solver
subprob_switch = 'linprog';
TRSP_options.solver = 'lhs';

load dfo.dat;

for row = [7, 43, 44]
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    L = -Inf(1, n);
    U = Inf(1, n);

    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    x0 = dfoxs(n, nprob, factor^factor_power)';
    problem = ['problem ' num2str(row) ' from More/Wild'];

    Options.hfun = @h_leastsquares;
    Options.combinemodels = @combine_leastsquares;
    Options.printf = 0;
    Options.spsolver = create_trsp_solver(spsolver);

    [~, ~, h_pounders, flag] = pounders(@(x)Ffun(x)', x0, n, nf_max, g_tol, delta_0, m, L, U, [], Options);
    assert(flag ~= -1, "pounders failed on " + problem);
    decrease_pounders = h_pounders(1) - min(h_pounders);
    assert(decrease_pounders > 0, "pounders did not decrease h on " + problem);

    rand('seed', 1);
    [X, ~, h, ~] = goombah(@h_leastsquares, Ffun, nf_max, x0, L, U, TRSP_options, subprob_switch);
    assert(size(X, 1) <= nf_max, "goombah grew the size of X on " + problem);
    assert(h(1) - min(h) >= 0.9 * decrease_pounders, "goombah decreased h much less than pounders on " + problem);

    rand('seed', 1);
    [X, ~, h, ~] = goombah_wo_msp(@h_leastsquares, Ffun, nf_max, x0, L, U, TRSP_options);
    assert(size(X, 1) <= nf_max, "goombah_wo_msp grew the size of X on " + problem);
    assert(h(1) - min(h) >= 0.9 * decrease_pounders, "goombah_wo_msp decreased h much less than pounders on " + problem);
end

path(oldpath);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
