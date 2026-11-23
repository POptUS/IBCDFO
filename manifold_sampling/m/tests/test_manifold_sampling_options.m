% This tests that manifold_sampling_primal accepts an Options struct and a
% Prior struct, and rejects ones that are not valid.

function [] = test_manifold_sampling_options()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

n = 3;
x0 = [10, 20, 30];
Low = -Inf(1, n);
Upp = Inf(1, n);
hfun = @h_pw_maximum_squared;
Ffun = @(x)x;
subprob_switch = 'linprog';
nf_max = 30;

% Options must be a struct
try
    manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, 'bad');
    error("Should have errored");
catch ME
    assert(strcmp(ME.message, "Options must be a struct"), "Unexpected error message: " + string(ME.message));
end

% Omitting Options and passing [] are equivalent
[X1, ~, ~, ~, ~] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch);
[X2, ~, ~, ~, ~] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, []);
assert(isequal(X1, X2), "Omitting Options and passing [] should give the same result");

% printf = 0 is silent
Options.printf = 0;
out = evalc('manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, Options);');
assert(isempty(out), "printf = 0 should produce no output");

% delta_0 sets the initial trust-region radius
Options.delta_0 = 0.5;
[X, ~, ~, ~, ~] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, Options);
assert(max(abs(X(2, :) - x0)) <= 0.5 + 1e-12, "Options.delta_0 was not used");

% Exhausting the budget gives flag 1
Options = struct('printf', 0);
[~, ~, ~, ~, flag] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, n + 2, subprob_switch, Options);
assert(flag == 1, "flag should be 1 when the budget is exhausted");

% Exit flags 0 and -6. Setting delta_min = delta_0 means that the main loop is
% never entered, so chi_k is never computed.
Options = struct('printf', 0, 'delta_0', 0.1, 'delta_min', 0.1);
[~, ~, ~, ~, flag] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, Options);
assert(flag == -6, "flag should be -6 when delta_min is reached without satisfying g_tol");
Options.g_tol = Inf; % Any stationary measure satisfies this
[~, ~, ~, ~, flag] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, Options);
assert(flag == 0, "flag should be 0 when delta_min is reached and g_tol is satisfied");

% Prior: a second run starts from the points of a first run and only
% evaluates nf_max new points
Options = struct('printf', 0);
[X1, F1, hF1, ~, ~] = manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, n + 5, subprob_switch, Options);
[~, i_best] = min(hF1);
Prior = struct('nfs', size(X1, 1), 'X_init', X1, 'F_init', F1, 'xk_in', i_best);
[X2, F2, ~, ~, ~] = manifold_sampling_primal(hfun, Ffun, X1(i_best, :), Low, Upp, 5, subprob_switch, Options, Prior);
assert(isequal(X2(1:Prior.nfs, :), X1) && isequal(F2(1:Prior.nfs, :), F1), "The Prior points should start X and F");
assert(size(X2, 1) <= Prior.nfs + 5, "manifold_sampling_primal evaluated more than nf_max new points");

% Bad Prior
assert_errors(@()manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch, Options, 'bad'), "Prior must be a struct");
assert_errors(@()manifold_sampling_primal(hfun, Ffun, x0 + 1, Low, Upp, nf_max, subprob_switch, Options, Prior), "x0 must equal X_init(xk_in, :)");
Prior_repeated = struct('nfs', Prior.nfs + 1, 'X_init', [X1; X1(1, :)], 'F_init', [F1; F1(1, :)], 'xk_in', i_best);
assert_errors(@()manifold_sampling_primal(hfun, Ffun, X1(i_best, :), Low, Upp, nf_max, subprob_switch, Options, Prior_repeated), "X_init contains repeated points");
Prior_bad_nfs = Prior;
Prior_bad_nfs.nfs = Prior.nfs - 1;
assert_errors(@()manifold_sampling_primal(hfun, Ffun, X1(i_best, :), Low, Upp, nf_max, subprob_switch, Options, Prior_bad_nfs), "Prior.F_init must have Prior.nfs rows");

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
