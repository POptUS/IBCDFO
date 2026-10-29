% Smooth sum-of-squares h function, following the [h, grads, Hash] convention of
% the manifold sampling h functions. Note there is always only one active
% "manifold" since h is smooth.

function [h, grads, Hash] = h_sum_squared(z, H0)
    z = z(:);
    if nargin == 1
        h = sum(z.^2);
        grads = 2 * z;
        Hash = {'1'};
    elseif nargin == 2
        J = length(H0);
        h = repmat(sum(z.^2), 1, J);
        grads = repmat(2 * z, 1, J);
    end
end
