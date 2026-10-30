% This tests that goombah and goombah_wo_msp run to termination, as opposed to
% exhausting their evaluation budget, on a problem that is easy. Row 1 of
% the More and Wild SIOPT benchmark "Benchmarking derivative-free optimization
% algorithms" is linear in x, so the component models are exact, and
% h(F(x)) = sum(F(x).^2) has the known minimum m - n = 36. The LHS TRSP solver is
% used instead of GAMS.
%
% To run this test, you must first install a BenDFO and a MINQ clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
%    /path/to/MINQ/m/minq5
% to the MATLAB path.

function [] = test_goombah_converge()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'tests', 'general_smooth_h_funs'));

% Once the trust region radius is tiny, fmincon warns that its finite-difference
% step does not fit within the bounds.
oldwarn = warning('off', 'optimlib:fwdFinDiffInsideBnds:StepReduced');

factor = 10;
nf_max = 400;
subprob_switch = 'linprog';
TRSP_options.solver = 'lhs';

load dfo.dat;

row = 1;
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

known_min = m - n;

rand('seed', 1);
[X, ~, h, ~] = goombah(@h_leastsquares, Ffun, nf_max, x0, L, U, TRSP_options, subprob_switch);
assert(size(X, 1) < nf_max, "goombah used its entire evaluation budget");
assert(abs(min(h) - known_min) <= 1e-6, "goombah did not converge to the known minimum");

rand('seed', 1);
[X, ~, h, ~] = goombah_wo_msp(@h_leastsquares, Ffun, nf_max, x0, L, U, TRSP_options);
assert(size(X, 1) < nf_max, "goombah_wo_msp used its entire evaluation budget");
assert(abs(min(h) - known_min) <= 1e-6, "goombah_wo_msp did not converge to the known minimum");

warning(oldwarn);
path(oldpath);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
