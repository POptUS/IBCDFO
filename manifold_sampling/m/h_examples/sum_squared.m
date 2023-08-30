function [h, grads, Hash] = sum_squared(z, H0)
% Evaluates the sum-squared objective
%   min sum_j { z_j^2 }
%
% Inputs:
%  z:              [1 x p]   point where we are evaluating h
%  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate z

% Outputs:
%  h: [dbl]                       function value
%  grads: [p x l]                 gradients of each of the l manifolds active at z
%  Hash: [1 x l cell of strings]  set of hashes for each of the l manifolds active at z (in the same order as the elements of grads)

z = z(:);
n = length(z);

if nargin == 1

    h = sum(z.^2);

    grads = zeros(n, 1);
    Hash = cell(1, 1);
    Hash{1} = int2str(1);
    grads(:) = 2 * z;

elseif nargin == 2
    h = sum(z.^2);
    grads = zeros(length(z), 1);
    grads(:) = 2 * z;
else
    error('Too many inputs to function');
end
