function [h, grads, Hash] = piecewise_quadratic(z, H0)

% Evaluates the piecewise quadratic function
%   max_j { || z - z_j ||_{Q_j}^2 + b_j }
%
% Inputs:
%  z:              [1 x p]   point where we are evaluating h
%  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

% Outputs:
%  h: [dbl]                       function value
%  grads: [p x l]                 gradients of each of the l quadratics active at z
%  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

% Hashes are output (and must be input) in the following fashion:
%   Hash{i} = 'j' if quadratic j is active at z (or H0{i} = 'j' if the
%   value/gradient of quadratic j at z is desired)

global Qs zs cs h_activity_tol

if isempty(h_activity_tol)
    h_activity_tol = 0;
end

z = z(:);

if nargin == 1
    [n, J] = size(zs);
    manifolds = zeros(1, J);
    for j = 1:J
        manifolds(j) = (z - zs(:, j))' * Qs(:, :, j) * (z - zs(:, j)) + cs(j);
    end

    h = max(manifolds);

    atol = h_activity_tol;
    rtol = h_activity_tol;
    inds = find(abs(h - manifolds) <= atol + rtol * abs(manifolds));

    grads = zeros(n, length(inds));

    Hash = cell(1, length(inds));
    for j = 1:length(inds)
        Hash{j} = int2str(inds(j));
        grads(:, j) = 2 * Qs(:, :, inds(j)) * (z - zs(:, inds(j)));
    end

elseif nargin == 2
    J = length(H0);
    h = zeros(1, J);
    grads = zeros(length(z), J);

    for k = 1:J
        j = str2num(H0{k});
        h(k) = (z - zs(:, j))' * Qs(:, :, j) * (z - zs(:, j)) + cs(j);
        grads(:, k) = 2 * Qs(:, :, j) * (z - zs(:, j));
    end

else
    error('Too many inputs to function');
end
