% This tests that goombah and goombah_wo_msp terminate gracefully, or error with
% an informative message, when given a bad objective, a bad option, or an
% evaluation budget that is exhausted while building the first models.
%
% To run this test, you must first install a MINQ clone and add
%    /path/to/MINQ/m/minq5
% to the MATLAB path.

function [] = test_goombah_failures()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'goombah', 'm', 'subproblems'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));
addpath(fullfile(repo_root, 'pounders', 'm'));

n = 3;
x0 = [10, 20, 30];
L = -Inf(1, n);
U = Inf(1, n);
hfun = @h_pw_maximum_squared;
Ffun = @(x)x;
TRSP_options.solver = 'lhs';
subprob_switch = 'linprog';

rand('seed', 1);

% Intentionally not passing a function for an objective
[X, F, h, xkin] = goombah(hfun, x0, 100, x0, L, U, TRSP_options, subprob_switch);
assert(isempty(X) && isempty(F) && isempty(h) && isempty(xkin), "Should have failed");

[X, F, h, xkin] = goombah_wo_msp(hfun, x0, 100, x0, L, U, TRSP_options);
assert(isempty(X) && isempty(F) && isempty(h) && isempty(xkin), "Should have failed");

% Supplying only enough evaluation budget for the initial simplex and one more
nf_max = n + 2;
[X, ~, ~, ~] = goombah(hfun, Ffun, nf_max, x0, L, U, TRSP_options, subprob_switch);
assert(size(X, 1) <= nf_max, "goombah grew the size of X");

[X, ~, ~, ~] = goombah_wo_msp(hfun, Ffun, nf_max, x0, L, U, TRSP_options);
assert(size(X, 1) <= nf_max, "goombah_wo_msp grew the size of X");

% L = U implies that no model can be built in any coordinate direction
nf_max = 100;
L_fixed = L;
U_fixed = U;
L_fixed(1) = x0(1);
U_fixed(1) = x0(1);
[X, F, h, ~] = goombah(hfun, Ffun, nf_max, x0, L_fixed, U_fixed, TRSP_options, subprob_switch);
assert(size(X, 1) < nf_max && size(F, 1) == size(X, 1) && numel(h) == size(X, 1), "goombah should have stopped early with consistent outputs");

[X, F, h, ~] = goombah_wo_msp(hfun, Ffun, nf_max, x0, L_fixed, U_fixed, TRSP_options);
assert(size(X, 1) < nf_max && size(F, 1) == size(X, 1) && numel(h) == size(X, 1), "goombah_wo_msp should have stopped early with consistent outputs");

% Bad options
bad_solver.solver = 'bad';
solver_message = "TRSP_options.solver must be 'GAMS' or 'lhs'";
assert_errors(@()goombah(hfun, Ffun, 100, x0, L, U, bad_solver, subprob_switch), solver_message);
assert_errors(@()goombah_wo_msp(hfun, Ffun, 100, x0, L, U, bad_solver), solver_message);
assert_errors(@()goombah_wo_msp(hfun, Ffun, 100, x0, L, U, TRSP_options, 'not a struct'), "Options must be a struct");

% The solver intended only for testing and debugging warns
oldwarn = warning('error', 'GOOMBAH:lhsTrspSolver');
lhs_message = "The lhs TRSP solver should only be used for testing or debugging";
assert_errors(@()goombah(hfun, Ffun, 100, x0, L, U, TRSP_options, subprob_switch), lhs_message);
assert_errors(@()goombah_wo_msp(hfun, Ffun, 100, x0, L, U, TRSP_options), lhs_message);
warning(oldwarn);

path(oldpath);
end

function [] = assert_errors(f, expected_message)
    try
        f();
    catch ME
        assert(strcmp(ME.message, expected_message), "Unexpected error message: " + string(ME.message));
        return
    end
    error("Should have errored");
end
