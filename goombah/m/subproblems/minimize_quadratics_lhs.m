% A GAMS-free solver for the GOOMBAH TRSP.
%
% This function
% - Forms the composed objective h_fun(z(s)), where z(s) are the `P` quadratic
%   functions of the step s := x - x0
% - Evaluates that objective on a Latin hypercube sample of [Lows, Upps], together
%   with the zero step and the step x1 - x0
% - Runs fmincon from the best of those points
% - Returns the resulting solution (the trial step) and its predicted decrease
%
% This method does not exploit the composite structure of the TRSP and is intended for
% testing GOOMBAH where GAMS is unavailable.

%%%%%%%%%%%
% Inputs:
%%%%%%%%%%%
% Hres:       [n-by-n-by-P] array of P model quadratic terms
% Gres:       [n-by-P] array of P model linear terms
% b:          [1-by-P] array of P model constant terms
% Lows:       [1-by-n] lower bounds on the step
% Upps:       [1-by-n] upper bounds on the step
% x0:         [1-by-n] trust region center
% x1:         [1-by-n] a candidate starting point
% val_at_x0:  [scalar] the objective value at x0
% h_fun:      [func handle] evaluates h

%%%%%%%%%%%
% Outputs:
%%%%%%%%%%%
% s_k:        [1-by-n] array containing the step
% pred_dec:   [scalar] predicted decrease from x0 to x0 + s_k

function [s_k, pred_dec] = minimize_quadratics_lhs(Hres, Gres, b, Lows, Upps, x0, x1, val_at_x0, h_fun)

[n, P] = size(Gres);

assert(all(size(Hres) == [n, n, P]), 'Hres must be n-by-n-by-P');
assert(all(size(b) == [1, P]), "b must be 1-by-P");
assert(all(size(Lows) == [1, n]) && all(size(Upps) == [1, n]), 'Lows and Upps must be 1-by-n');
assert(all(size(x0) == [1, n]) && all(size(x1) == [1, n]), 'x0 and x1 must be 1-by-n');
assert(all(Lows <= Upps), 'Lows must not exceed Upps');

obj = @(s) h_fun(eval_quadratics(Hres, Gres, b, s));

% Latin hypercube sample: each coordinate's num_samples values fall in
% distinct equal-width slices of [Lows(j), Upps(j)].
num_samples = 1000 * n;
S = zeros(num_samples, n);
for j = 1:n
    S(:, j) = Lows(j) + (Upps(j) - Lows(j)) * (randperm(num_samples)' - rand(num_samples, 1)) / num_samples;
end
S = [zeros(1, n); min(Upps, max(Lows, x1 - x0)); S];

vals = zeros(size(S, 1), 1);
for i = 1:size(S, 1)
    vals(i) = obj(S(i, :));
end
[~, ind] = min(vals);

options = optimoptions('fmincon', 'Display', 'off');
s_k = fmincon(obj, S(ind, :), [], [], [], [], Lows, Upps, [], options);
s_k = min(Upps, max(Lows, s_k));

% safeguard in case fmincon returns a worse solution than its starting point
if obj(s_k) > vals(ind)
    s_k = S(ind, :);
end

pred_dec = val_at_x0 - obj(s_k);

if pred_dec < 0
    pred_dec = 0;
end

end

function z = eval_quadratics(Hres, Gres, b, s)
P = size(Gres, 2);
z = zeros(1, P);
for i = 1:P
    z(i) = 0.5 * s * Hres(:, :, i) * s' + s * Gres(:, i) + b(i);
end
end
