% This tests that goombah terminates gracefully, or error with
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
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

n = 3;
x0 = [10, 20, 30];
Low = -Inf(1, n);
Upp = Inf(1, n);
hfun = @h_pw_maximum_squared;
Ffun = @(x)x;
TRSP_options.solver = 'lhs';
subprob_switch = 'linprog';

nf_max = 100; % Setting a budget that won't be used for the calls below that are expected to fail

rand('seed', 1);

% Intentionally not passing a function for an objective
[X, F, hF, xk_in] = goombah(hfun, x0, x0, Low, Upp, nf_max, TRSP_options, subprob_switch);
assert(isempty(X) && isempty(F) && isempty(hF) && isempty(xk_in), "Should have failed");

% Supplying only enough evaluation budget for the initial simplex and one more
nf_max_small = n + 2;
[X, ~, ~, ~] = goombah(hfun, Ffun, x0, Low, Upp, nf_max_small, TRSP_options, subprob_switch);
assert(size(X, 1) <= nf_max_small, "goombah grew the size of X");

% Low = Upp implies that no model can be built in any coordinate direction
Low_fixed = Low;
Upp_fixed = Upp;
Low_fixed(1) = x0(1);
Upp_fixed(1) = x0(1);
[X, F, hF, ~] = goombah(hfun, Ffun, x0, Low_fixed, Upp_fixed, nf_max, TRSP_options, subprob_switch);
assert(size(X, 1) < nf_max && size(F, 1) == size(X, 1) && numel(hF) == size(X, 1), "goombah should have stopped early with consistent outputs");

% Bad options
bad_solver.solver = 'bad';
solver_message = "TRSP_options.solver must be 'GAMS' or 'lhs'";
assert_errors(@()goombah(hfun, Ffun, x0, Low, Upp, nf_max, bad_solver, subprob_switch), solver_message);

% Exit flag: a failing objective gives -1 and an exhausted budget gives a
% positive flag (the final stationary measure)
[~, ~, ~, ~, flag] = goombah(hfun, x0, x0, Low, Upp, nf_max, TRSP_options, subprob_switch);
assert(flag == -1, "flag should be -1 when Ffun fails");
[X, F, hF, ~, flag] = goombah(hfun, Ffun, x0, Low, Upp, n + 5, TRSP_options, subprob_switch);
assert(flag > 0, "flag should be positive when the budget is exhausted");
assert(size(X, 1) == n + 5 && size(F, 1) == size(X, 1) && numel(hF) == size(X, 1), "outputs should have nf rows");

% The solver intended only for testing and debugging warns
oldwarn = warning('error', 'GOOMBAH:lhsTrspSolver');
lhs_message = "The lhs TRSP solver should only be used for testing or debugging";
assert_errors(@()goombah(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, subprob_switch), lhs_message);
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
