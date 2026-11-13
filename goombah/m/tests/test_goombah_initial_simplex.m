% This tests the n initial coordinate steps that goombah takes before its main
% loop. One coordinate is given to each of the four cases in the step choice:
%
%   1. room for +delta above the incumbent
%   2. no room above, but room for -delta below
%   3. box narrower than delta, with the wider side above
%   4. box narrower than delta, with the wider side below

function [] = test_goombah_initial_simplex()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

n = 4;
delta_0 = 0.1;

% Coordinate i exercises case i above. Coordinates 3 and 4 sit in a box of
% width 0.05, narrower than delta_0, so neither a +delta nor a -delta step fits.
x0 =  [0.0,  1.0,  0.01, 0.04];
Low = [-1.0, -1.0, 0.0,  0.0];
Upp = [1.0,  1.0,  0.05, 0.05];
expected_steps = [delta_0, -delta_0, 0.04, -0.04];

hfun = @h_pw_maximum_squared;
Ffun = @(x) x;
TRSP_options.solver = 'lhs';
Options = struct('printf', 0, 'delta_0', delta_0);

% A budget of n + 1 returns as soon as the initial simplex has been evaluated
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

% Whichever case applies, no requested point may leave the box
assert(all(all(X >= Low)) && all(all(X <= Upp)), "goombah requested a point outside the bounds");

path(oldpath);
end
