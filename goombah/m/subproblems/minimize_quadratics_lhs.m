% A GAMS-free solver for the GOOMBAH TRSP.
%
% This function
% - Forms the composed objective h_fun(z(s)), where z(s) are the `P` quadratics
%   in the step s := x - x0
% - Evaluates that objective on a Latin hypercube sample of [Low, Upp], together
%   with the zero step and the step x1 - x0
% - Starts fmincon from the best of those points
% - Returns the resulting step and its predicted decrease
%
% This does not exploit the composite structure of the TRSP and is intended for
% testing GOOMBAH where GAMS is unavailable.

%%%%%%%%%%%
% Inputs:
%%%%%%%%%%%
% H:          [n-by-n-by-P] array of P model quadratic terms
% g:          [n-by-P] array of P model linear terms
% b:          [P-by-1] array of P model constant terms
% Low:        [1-by-n] lower bounds on the step
% Upp:        [1-by-n] upper bounds on the step
% x0:         [1-by-n] trust region center
% x1:         [1-by-n] a candidate starting point
% val_at_x0:  [scalar] the objective value at x0
% h_fun:      [func handle] evaluates h

%%%%%%%%%%%
% Outputs:
%%%%%%%%%%%
% s_k:        [1-by-n] array containing the step
% pred_dec:   [scalar] predicted decrease from x0 to x0 + s_k

function [s_k, pred_dec] = minimize_quadratics_lhs(H, g, b, Low, Upp, x0, x1, val_at_x0, h_fun)

[n, P] = size(g);

assert(size(H, 1) == n && size(H, 2) == n && size(H, 3) == P, 'H must be n-by-n-by-P');
assert(numel(b) == P, 'b must have P entries');
assert(all(size(Low) == [1, n]) && all(size(Upp) == [1, n]), 'Low and Upp must be 1-by-n');
assert(all(size(x0) == [1, n]) && all(size(x1) == [1, n]), 'x0 and x1 must be 1-by-n');
assert(all(Low <= Upp), 'Low must not exceed Upp');

obj = @(s) h_fun(eval_quadratics(H, g, b, s));

% Latin hypercube sample: each coordinate's num_samples values fall in
% distinct equal-width slices of [Low(j), Upp(j)].
num_samples = 1000 * n;
S = zeros(num_samples, n);
for j = 1:n
    S(:, j) = Low(j) + (Upp(j) - Low(j)) * (randperm(num_samples)' - rand(num_samples, 1)) / num_samples;
end
S = [zeros(1, n); min(Upp, max(Low, x1 - x0)); S];

vals = zeros(size(S, 1), 1);
for i = 1:size(S, 1)
    vals(i) = obj(S(i, :));
end
[~, ind] = min(vals);

options = optimoptions('fmincon', 'Display', 'off');
s_k = fmincon(obj, S(ind, :), [], [], [], [], Low, Upp, [], options);
s_k = min(Upp, max(Low, s_k));

% fmincon may end worse than its (nonsmooth) starting point
if obj(s_k) > vals(ind)
    s_k = S(ind, :);
end

pred_dec = val_at_x0 - obj(s_k);

if pred_dec < 0
    pred_dec = 0;
end

end

function z = eval_quadratics(H, g, b, s)
P = size(g, 2);
z = zeros(1, P);
for i = 1:P
    z(i) = 0.5 * s * H(:, :, i) * s' + s * g(:, i) + b(i);
end
end
