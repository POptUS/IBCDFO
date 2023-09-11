function [h, grads, Hash] = quantile(z, H0)
% Evaluates the q^th quantile of the values
%  { z_j^2 }
%
% Inputs:
%  z:              [1 x p]   point where we are evaluating h
%  H0: (optional)  [1 x l cell of strings]  set of hashes where to evaluate

% Outputs:
%  h: [dbl]                       function value
%  grads: [p x l]                 gradients of each of the l quadratics active at z
%  Hash: [1 x l cell of strings]  set of hashes for each of the l quadratics active at z (in the same order as the elements of grads)

q = 2; % hard-coded. Eventually, make global or (preferred) pass it in as an argument.

z = z(:);
n = length(z);
z2 = z.^2;
[sortedz2, sort_inds] = sort(z2);

if nargin == 1
    h = sortedz2(q);

    atol = 1e-8;
    rtol = 1e-8;
    inds = find(abs(h - z2) <= atol + rtol * abs(z2));

    grads = zeros(n, length(inds));

    Hash = cell(1, length(inds));
    for j = 1:length(inds)
        Hash{j} = int2str(inds(j));
        grads(inds(j), j) = 2 * z(inds(j));
    end

elseif nargin == 2
    J = length(H0);
    h = zeros(1, J);
    grads = zeros(length(z), J);
    z = z(sort_inds);

    for k = 1:J
        j = str2num(H0{k});
        h(k) = z(j)^2;
        grads(j, k) = 2 * z(j);
    end
end
