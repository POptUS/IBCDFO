% This tests the n initial evaluations that goombah requests before its main
% loop. Each coordinate tests each of one of these four cases:
%
%   1. [x0]_i +/- delta are both feasible (that is, in [Low]_i, [Upp]_i)
%   2. [x0]_i + delta not feasible, but [x0]_i - delta is feasible
%   3. Neither [x0]_i +/- delta is feasible, right hand side wider
%   4. Neither [x0]_i +/- delta is feasible, left hand side wider

function [] = test_goombah_initial_simplex()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

n = 4;
delta_0 = 0.1;

% Coordinate i tests case i from comments. Coordinates 3 and 4 sit in a box of
% width 0.05, narrower than delta_0, so neither +/- delta are feasible.
x0 =  [0.0,  1.0,  0.01, 0.04];
Low = [-1.0, -1.0, 0.0,  0.0];
Upp = [1.0,  1.0,  0.05, 0.05];
expected_steps = [delta_0, -delta_0, 0.04, -0.04];

hfun = @h_pw_maximum_squared;
Ffun = @(x) x;
TRSP_options.solver = 'lhs';
Options = struct('printf', 0, 'delta_0', delta_0);

% A budget of n + 1 forces a return right after the initial simplex has been evaluated
[X, ~, ~, ~, flag] = goombah(hfun, Ffun, x0, Low, Upp, n + 1, TRSP_options, 'linprog', Options);

assert(flag == 1, "goombah should stop once the budget is spent on the initial simplex");
assert(isequal(size(X), [n + 1, n]), "goombah should evaluate x0 and one point per coordinate");
assert(isequal(X(1, :), x0), "the first evaluated point should be x0");

for i = 1:n
    step = X(i + 1, :) - x0;
    others = setdiff(1:n, i);
    assert(all(step(others) == 0), sprintf("point %d should differ from x0 in coordinate %d alone", i + 1, i));
    assert(abs(step(i) - expected_steps(i)) <= 1e-12, ...
           sprintf("coordinate %d stepped %g, expected %g", i, step(i), expected_steps(i)));
end

% Whichever case applies, no requested point can be infeasible
assert(all(all(X >= Low)) && all(all(X <= Upp)), "goombah requested a point outside the bounds");

path(oldpath);
end
