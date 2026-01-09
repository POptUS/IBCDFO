function [h, grads, Hash] = h_piecewise_quadratic_1(z, H0)
% Evaluates the piecewise quadratic function
%   max_j { 0.5*z'*Qs_j*z + zs_j'*z + cs_j }
%
% Inputs:
%  z:              [1 x p]   point where we are evaluating h
%  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

% Outputs:
%  h: [dbl]                       function value
%  grads: [p x l]                 gradients of each of the l quadratics active at z
%  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

global Qs zs cs

z = z(:);

if nargin == 1
    [n, J] = size(zs);
    manifolds = zeros(1, J);
    for j = 1:J
        manifolds(j) = 0.5 * z' * Qs(:, :, j) * z + zs(:, j)' * z + cs(j);
    end

    h = max(manifolds);

    atol = 1e-12;
    rtol = 1e-12;
    inds = find(abs(h - manifolds) <= atol + rtol * abs(h - manifolds));

    grads = zeros(n, length(inds));

    Hash = cell(1, length(inds));
    for j = 1:length(inds)
        Hash{j} = int2str(inds(j));
        grads(:, j) = Qs(:, :, inds(j)) * z + zs(:, inds(j));
    end

elseif nargin == 2
    J = length(H0);
    h = zeros(1, J);
    grads = zeros(length(z), J);

    for k = 1:J
        j = str2num(H0{k});
        h(k) = 0.5 * z' * Qs(:, :, j) * z + zs(:, j)' * z + cs(j);
        grads(:, k) = Qs(:, :, j) * z + zs(:, j);
    end
end
