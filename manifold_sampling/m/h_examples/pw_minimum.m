function [h, grads, Hash] = pw_minimum(z, H0)
% Evaluates the pointwise maximum function
%   min_j { z_j }
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

    h = min(z);

    atol = 1e-8;
    rtol = 1e-8;
    inds = find(abs(h - z) <= atol + rtol * abs(z));

    grads = zeros(n, length(inds));

    Hash = cell(1, length(inds));
    for j = 1:length(inds)
        Hash{j} = int2str(inds(j));
        grads(inds(j), j) = 1;
    end

elseif nargin == 2
    J = length(H0);
    h = zeros(1, J);
    grads = zeros(length(z), J);

    for k = 1:J
        j = str2num(H0{k});
        h(k) = z(j);
        grads(j, k) = 1;
    end
end
