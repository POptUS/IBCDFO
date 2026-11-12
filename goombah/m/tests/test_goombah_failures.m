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

% Low = Upp leaves a coordinate with no room to build a model in, so the input
% checks reject it
Low_fixed = Low;
Upp_fixed = Upp;
Low_fixed(1) = x0(1);
Upp_fixed(1) = x0(1);
[X, F, hF, xk_in, flag] = goombah(hfun, Ffun, x0, Low_fixed, Upp_fixed, nf_max, TRSP_options, subprob_switch);
assert(flag == -1, "goombah should have rejected Low == Upp");
assert(isempty(X) && isempty(F) && isempty(hF) && isempty(xk_in), "a rejected problem should return empty outputs");

% A starting point outside the bounds is rejected too
[X, F, hF, xk_in, flag] = goombah(hfun, Ffun, x0, Low, x0 - 1, nf_max, TRSP_options, subprob_switch);
assert(flag == -1, "goombah should have rejected an x0 above Upp");
assert(isempty(X) && isempty(F) && isempty(hF) && isempty(xk_in), "a rejected problem should return empty outputs");

% Bad options
bad_solver.solver = 'bad';
solver_message = "TRSP_options.solver must be 'GAMS' or 'lhs'";
assert_errors(@()goombah(hfun, Ffun, x0, Low, Upp, nf_max, bad_solver, subprob_switch), solver_message);

% Bad Options
assert_errors(@()goombah(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, subprob_switch, 'bad'), "Options must be a struct");

% Options: printf = 0 prints nothing
Options.printf = 0;
out = evalc('goombah(hfun, Ffun, x0, Low, Upp, n + 5, TRSP_options, subprob_switch, Options);');
assert(isempty(out), "printf = 0 should produce no output");

% Options: delta_0 sets the initial trust-region radius
Options.delta_0 = 0.5;
[X, ~, ~, ~] = goombah(hfun, Ffun, x0, Low, Upp, n + 2, TRSP_options, subprob_switch, Options);
assert(max(abs(X(2, :) - x0)) == 0.5, "Options.delta_0 was not used for the initial simplex");

% The following covers beta_exp and tests that there is an actual
% effect. Both runs use the same seed and budget, so they evaluate the same
% n + 2 points and differ only after that test is made.
x0_near = [0.03, 0.04, 0.05];
Options_beta = struct('printf', 0, 'beta_exp', 1.0);
rand('seed', 1);
[X_accepted, ~, hF_accepted] = goombah(hfun, Ffun, x0_near, Low, Upp, n + 3, TRSP_options, subprob_switch, Options_beta);
Options_beta.beta_exp = -1.0;
rand('seed', 1);
[X_rejected, ~, hF_rejected] = goombah(hfun, Ffun, x0_near, Low, Upp, n + 3, TRSP_options, subprob_switch, Options_beta);
assert(isequal(X_accepted(1:n + 2, :), X_rejected(1:n + 2, :)), "The two runs should agree until the first GOOMBAH step is accepted or rejected");
assert(~isequal(X_accepted(n + 3, :), X_rejected(n + 3, :)), "Options.beta_exp was not used when deciding whether to take the GOOMBAH step");
assert(min(hF_accepted) < min(hF_rejected), "Accepting the GOOMBAH step should find a better point within the same budget");

% Exit flags 0 and -6. Setting delta_min = delta_0 means that the main loop is
% never entered, so no trust-region subproblem is solved and chi_k is never computed.
Options_flags = struct('printf', 0, 'delta_0', 0.1, 'delta_min', 0.1);
[~, ~, ~, ~, flag] = goombah(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, subprob_switch, Options_flags);
assert(flag == -6, "flag should be -6 when delta_min is reached without satisfying g_tol");
Options_flags.g_tol = Inf; % Any stationary measure satisfies this
[~, ~, ~, ~, flag] = goombah(hfun, Ffun, x0, Low, Upp, nf_max, TRSP_options, subprob_switch, Options_flags);
assert(flag == 0, "flag should be 0 when delta_min is reached and g_tol is satisfied");

% The following covers the convergence test within the MS-P loop. g_tol = Inf
% means every point is a stationary point. Because delta_min is between
% delta_0 and gamma_dec * delta_0, the first time the MS-P loop shrinks
% delta, the loop returns with a flag of 0. h_pw_minimum_squared is used
% because a selection function that is active at a nearby point can be
% inactive at the trust-region center, which is what triggers the delta
% shrink within the MS-P loop.
Options_msp = struct('printf', 0, 'g_tol', Inf, 'delta_0', 0.1, 'delta_min', 0.09);
rand('seed', 1);
[X_msp, ~, ~, ~, flag] = goombah(@h_pw_minimum_squared, Ffun, x0_near, Low, Upp, nf_max, TRSP_options, subprob_switch, Options_msp);
assert(flag == 0, "flag should be 0 when the MS-P loop meets both the stationarity and the radius test");
assert(size(X_msp, 1) < nf_max, "goombah should have returned from the MS-P loop well before exhausting its budget");

% The following covers the budget check within the MS-P loop.
% The same h and starting point as the prior test, but with the default radii.
% This takes a second pass through the MS-P loop, but the budget
% nf_max =n + 4 is exhausted by model building in the second pass, so goombah
% returns a flag of 1.
rand('seed', 1);
[X_msp, F_msp, hF_msp, ~, flag] = goombah(@h_pw_minimum_squared, Ffun, x0_near, Low, Upp, n + 4, TRSP_options, subprob_switch, struct('printf', 0));
assert(flag == 1, "flag should be 1 when the MS-P loop exhausts the budget while building its models");
assert(size(X_msp, 1) == n + 4 && size(F_msp, 1) == n + 4 && numel(hF_msp) == n + 4, "outputs should have nf rows");

% Exit flag: a failing objective gives -1 and an exhausted budget gives a
% flag of 1
[~, ~, ~, ~, flag] = goombah(hfun, x0, x0, Low, Upp, nf_max, TRSP_options, subprob_switch);
assert(flag == -1, "flag should be -1 when Ffun fails");
[X, F, hF, ~, flag] = goombah(hfun, Ffun, x0, Low, Upp, n + 5, TRSP_options, subprob_switch);
assert(flag == 1, "flag should be 1 when the budget is exhausted");
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
