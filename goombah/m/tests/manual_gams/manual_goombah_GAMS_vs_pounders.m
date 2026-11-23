% MANUAL check of goombah+GAMS against pounders on least-squares problems
% h(F(x)) = sum(F(x).^2) from the More and Wild SIOPT paper "Benchmarking
% derivative-free optimization algorithms". Least squares is the only h
% function tested to which pounders applies, so it is the one place this
% GAMS-based TRSP flow can be checked against an independent solver rather
% than only against the lhs TRSP solver (see manual_goombah_gams_check.m).
%
% This is NOT part of the automatic test suite (Testgoombah, CI): GAMS is
% licensed software that is unavailable on GitHub runners (Issue 332). Its
% filename deliberately does not begin or end with "test" so that runtests
% does not collect it. Reviewers with a GAMS installation should run it by
% hand when a PR changes GOOMBAH or its GAMS subproblems.
%
% Requirements:
%    * the gams executable on the system path
%    * GDXMRW (wgdx/rgdx) on the MATLAB path. Recent GAMS releases may not ship
%      the MATLAB MEX files; if yours does not, add the GDXMRW folder of an
%      older GAMS installation (the MEX files from an older release worked
%      with a newer gams executable).
%    * BenDFO (https://github.com/POptUS/BenDFO: its data and m folders) and
%      MINQ (https://github.com/POptUS/minq: m/minq5, at the commit in
%      REQUIRED_MINQ_COMMIT in the IBCDFO repository root) on the MATLAB path
%
% From MATLAB:
%     >> addpath('/path/to/IBCDFO/goombah/m/tests/manual_gams')
%     >> manual_goombah_GAMS_vs_pounders
%
% For each problem, pounders is run to build a reference decrease in h, then
% goombah is run with the GAMS TRSP solver from the same starting point; the
% GAMS run must decrease h at least 99.9% as much as pounders does. Test
% execution terminates if any errors or failures occur.  Reviewers may wish to
% note the result in the PR (no results are reported back automatically).

function [] = manual_goombah_GAMS_vs_pounders()

assert(system('which gams > /dev/null') == 0, ...
       'The gams executable is not on the system path');
assert(exist('wgdx', 'file') > 0, 'GDXMRW (wgdx) is not on the MATLAB path');

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'pounders', 'm'));
addpath(fullfile(repo_root, 'pounders', 'm', 'general_h_funs'));
% h_leastsquares is used both by goombah (which needs its 3-output,
% manifold-sampling-style interface) and by pounders (which only ever calls it
% for a scalar), so use the one version that supports both; added last so it
% takes precedence over pounders/m/general_h_funs's single-output version.
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'tests', 'general_smooth_h_funs'));
trsp_root = fullfile(repo_root, 'goombah', 'm', 'subproblems');

factor = 10;
nf_max = 30;
g_tol = 1e-13;
delta_0 = 0.1;
subprob_switch = 'linprog';
GAMS_options = struct('solver', 'GAMS', ...
                       'file', fullfile(trsp_root, 'minimize_sum_quad_models_squared.gms'), ...
                       'solvers', 1:4);
Options_pounders = struct('hfun', @h_leastsquares, 'combinemodels', @combine_leastsquares, ...
                          'printf', 0, 'spsolver', create_trsp_solver(2));

load dfo.dat;

for row = 1
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    LB = -5000 * ones(1, n);
    UB = 5000 * ones(1, n);
    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    x0_prob = dfoxs(n, nprob, factor^factor_power)';
    problem = ['problem ' num2str(row) ' with hfun=h_leastsquares'];
    fprintf('Running %s\n', problem);

    [~, ~, h_pounders, flag, xkin_pounders] = pounders(@(x)Ffun(x)', x0_prob, n, nf_max, g_tol, delta_0, m, LB, UB, [], Options_pounders);
    assert(flag > 0, "pounders ended with flag = " + flag);
    decrease_pounders = h_pounders(1) - h_pounders(xkin_pounders);
    assert(decrease_pounders > 0, "pounders did not decrease h");

    rand('seed', 1);
    [X, F, h, xkin, flag] = goombah(@h_leastsquares, Ffun, x0_prob, LB, UB, nf_max, subprob_switch, struct('TRSP', GAMS_options));
    assert(flag == 1, "goombah with GAMS ended with flag = " + flag);
    assert(size(X, 1) == nf_max, "goombah with GAMS did not terminate at evaluation budget");
    assert(size(X, 1) == size(F, 1) && size(F, 1) == numel(h), "goombah with GAMS has inconsistent output sizes");
    decrease_goombah = h(1) - h(xkin);
    assert(decrease_goombah > 0, "goombah with GAMS did not decrease");
    assert(decrease_goombah >= 0.999 * decrease_pounders, "goombah GAMS decreased h much less than pounders");

    fprintf('  h(xkin): GAMS %g, pounders %g\n', h(xkin), h_pounders(xkin_pounders));
end

fprintf('All manual GAMS checks passed.\n');
path(oldpath);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
