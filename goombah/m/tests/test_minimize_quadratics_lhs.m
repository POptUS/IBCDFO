% This tests the LHS-based GOOMBAH TRSP solver on an instance with a known
% minimizer: the linear models z(s) = s - c have a common zero at s = c, which
% is inside the box, so the minimum of max(z.^2) is 0.

function [] = test_minimize_quadratics_lhs()

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..', 'subproblems'));
addpath(fullfile(here_path, '..', '..', '..', 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

rng(1);

n = 3;
c = [0.1, -0.2, 0.3];
H = zeros(n, n, n);
g = eye(n);
b = -c';
Low = -0.5 * ones(1, n);
Upp = 0.5 * ones(1, n);
x0 = ones(1, n);

val_at_x0 = h_pw_maximum_squared(b);

[s_k, pred_dec] = minimize_quadratics_lhs(H, g, b, Low, Upp, x0, x0, val_at_x0, @h_pw_maximum_squared);

assert(all(Low <= s_k) && all(s_k <= Upp), "Step is not within the bounds");
assert(abs(pred_dec - val_at_x0) <= 1e-6, "Did not find the known minimum");

path(oldpath);
end
