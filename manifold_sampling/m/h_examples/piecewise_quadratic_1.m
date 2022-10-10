function [h, grads, Hash] = piecewise_quadratic_1(z, H0)
% Evaluates the piecewise quadratic function
%   max_j { 0.5*z'*Q_j*z + c_j'*z + b_j }
%
% Inputs:
%  z:              [1 x p]   point where we are evaluating h
%  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

% Outputs:
%  h: [dbl]                       function value
%  grads: [p x l]                 gradients of each of the l quadratics active at z
%  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

z = z(:);
global Q c b

if nargin == 1
    J = length(b);
    manifolds = zeros(1, J);
    for j = 1:J
        manifolds(j) = 0.5 * z' * Q(:, :, j) * z + c(:, j)' * z + b(j);
    end

    h = max(manifolds);

    atol = 1e-12;
    rtol = 1e-12;
    inds = find(abs(h - manifolds) <= atol + rtol * abs(h - manifolds));

    grads = Q(:, :, inds) * z + c(:, inds);

    Hash = cell(1, length(inds));
    for j = 1:length(inds)
        Hash{j} = int2str(inds(j));
    end

elseif nargin == 2
    J = length(H0);
    h = zeros(1, J);
    grads = zeros(length(z), J);

    for k = 1:J
        j = str2num(H0{k});
        h(k) = 0.5 * z' * Q(:, :, j) * z + c(:, j)' * z + b(j);
        grads(:, k) = Q(:, :, j) * z + c(:, j);
    end

else
    error('Too many inputs to function');
end
