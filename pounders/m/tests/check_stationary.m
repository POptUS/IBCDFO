% This function checks if a point x is approximately stationary for the
% objective defined by BenDFO and combinemodels.

function [] = check_stationary(xvec, L, U, BenDFO, combinemodels)
    n = BenDFO.n;
    m = BenDFO.m;
    [fvec, G] = calfun_jacobian_wrapper(xvec, BenDFO, 'smooth');

    grad = combinemodels(fvec(:)', G, zeros(n, n, m));

    ub_inds = xvec == U;
    lb_inds = xvec == L;
    grad(ub_inds) = max(0, grad(ub_inds));
    grad(lb_inds) = min(0, grad(lb_inds));
    assert(norm(grad) <= 1e-8);
end

function [fvec, J] = calfun_jacobian_wrapper(x, struct, probtype)
    [~, fvec, ~, J] = calfun(x, struct, probtype);
end
