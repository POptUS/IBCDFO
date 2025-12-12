% This tests pounders
%   - the spsolver=1 and spsolver=3 cases
%   - the handling of bounds
%   - printing
%   - Using starting points in X_0 and Ffun(X_0)
%
% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
% to the MATLAB path.

function [] = test_bounds_and_sp1()

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));
addpath(fullfile(here_path, '..', 'general_h_funs'));

nf_max = 100;
g_tol = 1e-13;
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

    np_max = 2 * n + 1;  % Maximum number of interpolation points [2*n+1]
    if row == 7
        Low = -2 * ones(1, n);
        Upp = 2 * ones(1, n);
    else
        Low = [-12, 0];
        Upp = [0, 10];
    end

    nfs = 1;
    X_0 = xs';
    xk_in = 1;
    delta_0 = 0.1;
    printf = 1;
    spsolver = 1;

    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    F_init = Ffun(X_0)';

    for spsolver = [1, 3]
        for hfun_cases = 1:3
            if hfun_cases == 1
                hfun = @(F)sum(F.^2);
                combinemodels = @combine_leastsquares;
            elseif hfun_cases == 2
                ALPHA = 0; % If changed here, also needs to be adjusted in squared_diff_from_mean.m
                hfun = @(F)sum((F - 1 / length(F) * sum(F)).^2) - ALPHA * (1 / length(F) * sum(F))^2;
                combinemodels = @(Cres, Gres, Hres) combine_squared_diff_from_mean(Cres, Gres, Hres, ALPHA);
            elseif hfun_cases == 3
                hfun = @(F)-1 * sum(F.^2);
                combinemodels = @combine_neg_leastsquares;
            end

            Prior.xk_in = xk_in;
            Prior.X_0 = X_0;
            Prior.F_init = F_init;
            Prior.nfs = nfs;

            Options.hfun = hfun;
            Options.combinemodels = combinemodels;
            Options.spsolver = spsolver;
            Options.printf = printf;

            [X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior, Options);

            if flag == 0
                check_stationary(X(xk_best, :), Low, Upp, BenDFO, combinemodels);
            end
        end
    end
end

% Test success without last (optional) arguments to pounders
[X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp);

path(oldpath);

assert(flag == 0, "Test didn't complete");
end

function [fvec] = calfun_wrapper(x, struct, probtype)
[~, fvec, ~] = calfun(x, struct, probtype);
end
