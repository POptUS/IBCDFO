% This tests pounders
%   - the spsolver=1 and spsolver=3 cases
%   - the handling of bounds
%   - printing
%   - Using starting points in X0 and F0

function [] = test_bounds_and_sp1()

nfmax = 100;
gtol = 1e-13;
factor = 10;

load dfo.dat;

for row = [7, 8]
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.m = m;
    BenDFO.n = n;

    xs = dfoxs(n, nprob, factor^factor_power);

    SolverNumber = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % POUNDERs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    npmax = 2 * n + 1;  % Maximum number of interpolation points [2*n+1]
    if row == 7
        L = -2 * ones(1, n);
        U = 2 * ones(1, n);
    else
        L = [-12, 0];
        U = [0, 10];
    end

    nfs = 1;
    X0 = xs';
    xkin = 1;
    delta = 0.1;
    printf = 1;
    spsolver = 1;

    objective = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    F0 = objective(X0)';

    for spsolver = [1, 3]
        for hfun_cases = 1:3
            if hfun_cases == 1
                hfun = @(F)sum(F.^2);
                combinemodels = @leastsquares;
            elseif hfun_cases == 2
                alpha = 0; % If changed here, also needs to be adjusted in squared_diff_from_mean.m
                hfun = @(F)sum((F - 1 / length(F) * sum(F)).^2) - alpha * (1 / length(F) * sum(F))^2;
                combinemodels = @squared_diff_from_mean;
            elseif hfun_cases == 3
                hfun = @(F)-1 * sum(F.^2);
                combinemodels = @neg_leastsquares;
            end

            [X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);

            if flag == 0
                check_stationary(X(xk_best, :), L, U, BenDFO, combinemodels);
            end
        end
    end
end

% Test success without last (optional) arguments to pounders
minq_location = '../../../../minq/m/minq5/';
addpath(minq_location);

[X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U);
assert(flag == 0, "Test didn't complete");
end

function [fvec] = calfun_wrapper(x, struct, probtype)
[~, fvec, ~] = calfun(x, struct, probtype);
end
