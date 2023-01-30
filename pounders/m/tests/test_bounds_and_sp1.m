% This tests pounders
%   - the spsolver=1 and spsolver=3 cases
%   - the handling of bounds
%   - printing
%   - Using starting points in X0 and F0

function [] = test_bounds_and_sp1()


addpath('../');
addpath('../general_h_funs');
bendfo_location = '../../../../BenDFO';
addpath([bendfo_location, '/m/']);
addpath([bendfo_location, '/data/']);
minq_location = '../../../../minq/m/minq8/';
addpath(minq_location);


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
        L = -2*ones(1, n);
        U = 2*ones(1, n);
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
            end

            [X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);
        end
    end
end
end

function [fvec] = calfun_wrapper(x, struct, probtype)
[~, fvec, ~] = calfun(x, struct, probtype);
end