% This tests that manifold_sampling_primal accepts an Options struct and
% rejects one that is not a struct.
%
% No external packages are needed.

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

path(oldpath);
end
