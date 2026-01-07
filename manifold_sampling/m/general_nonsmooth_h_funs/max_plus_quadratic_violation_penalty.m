function [h, grads, Hashes] = max_plus_quadratic_violation_penalty(z, H0)
% This the outer h function required by manifold sampling.
% If z \in R^p
% It encodes the objective
%    max_{i = 1,...,p1} z_i + alpha *sum_{i = p1+1}^{p} max(z_i, 0)^2
%
% Hashes are output (and must be input) in the following fashion:
%   Hash elements are strings of p integers.
%     0 in position 1 <= i <= p1 means max_{i = 1,...,p1} z_i > z_i
%     1 in position 1 <= i <= p1 means max_{i = 1,...,p1} z_i = z_i
%     0 in position p1+1 <= i <= p means max(z_i,0) = 0
%     1 in position p1+1 <= i <= p means max(z_i,0) = z_i
%
%   Similarly, if H0 has a 1 in position i uses z_i in the calculation of h and grads.

% Get data from outside of this function
p1 = length(z) - 1;
alpha = 0;
h_activity_tol = 1e-8;

p = length(z);

if nargin == 1

    h1 = max(z(1:p1));
    h2 = alpha * sum(max(z(p1 + 1:end), 0).^2);
    h = h1 + h2;

    atol = h_activity_tol;
    rtol = h_activity_tol;

    inds1 = find(abs(h1 - z(1:p1)) <= atol + rtol * abs(z(1:p1)));
    inds2 = p1 + find(z(p1 + 1:end) >= -rtol);

    grads = zeros(p, length(inds1));
    Hashes = cell(1, length(inds1));

    for j = 1:length(inds1)
        hash = dec2bin(0, p);
        hash(inds1(j)) = '1';
        hash(inds2) = '1';
        Hashes{j} = hash;
        grads(inds1(j), j) = 1;
        grads(inds2, j) = alpha * 2 * z(inds2);
    end

elseif nargin == 2
    J = length(H0);
    h = zeros(1, J);
    grads = zeros(p, J);

    for k = 1:J
        max_ind = find(H0{k}(1:p1) == '1');
        assert(length(max_ind) == 1, "I don't know what to do in this case");
        grads(max_ind, k) = 1;

        h1 = z(max_ind);

        const_viol_inds = p1 + find(H0{k}(p1 + 1:end) == '1');
        if isempty(const_viol_inds)
            h2 = 0;
        else
            grads(const_viol_inds, k) = alpha * 2 * z(const_viol_inds);
            h2 = alpha * sum(max(z(const_viol_inds), 0).^2);
        end
        h(k) = h1 + h2;
    end
end
