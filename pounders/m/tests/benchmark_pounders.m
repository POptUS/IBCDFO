% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
%
% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
% to the MATLAB path.

function [] = benchmark_pounders()

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));
addpath(fullfile(here_path, '..', 'general_h_funs'));

load dfo.dat;

ensure_still_solve_problems = 0;
if ensure_still_solve_problems
    solved = load('./benchmark_results/solved.txt'); % A 0-1 matrix with 1 when problem was previously solved.
else
    solved = zeros(53, 3);
end

spsolver = 2; % TRSP Solver
nf_max = 100;
g_tol = 1e-13;
factor = 10;
nfs = 0;

for row = 1:length(dfo)
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.m = m;
    BenDFO.n = n;

    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');

    X_0 = dfoxs(n, nprob, factor^factor_power)';
    Low = -Inf(1, n);
    Upp = Inf(1, n);
    delta_0 = 0.1;
    if row == 9
        printf = 1;
    else
        printf = 0;
    end

    for hfun_cases = 1:3
        Results = cell(3, 53);
        if hfun_cases == 1
            hfun = @(F)sum(F.^2);
            combinemodels = @leastsquares;
        elseif hfun_cases == 2
            alpha = 0; % If changed here, also needs to be adjusted in squared_diff_from_mean.m
            hfun = @(F)sum((F - 1 / length(F) * sum(F)).^2) - alpha * (1 / length(F) * sum(F))^2;
            combinemodels = @squared_diff_from_mean;
        elseif hfun_cases == 3
            if m ~= 3 % Emittance is only defined for the case when m == 3
                continue
            end
            hfun = @emittance_h;
            combinemodels = @emittance_combine;
            printf = 2; % Just to test this feature
        end
        disp([row, hfun_cases]);

        filename = ['./benchmark_results/poundersM_nf_max=' int2str(nf_max) '_gtol=' num2str(g_tol) '_prob=' int2str(row) '_spsolver=' num2str(spsolver) '_hfun=' func2str(combinemodels) '.mat'];

        Options.hfun = hfun;
        Options.combinemodels = combinemodels;
        Options.printf = printf;

        [X, F, hF, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, [], Options);

        if ensure_still_solve_problems
            if solved(row, hfun_cases) == 1
                assert(flag == 0, "This problem was previously solved but it's anymore.");
                check_stationary(X(xk_best, :), L, U, BenDFO, combinemodels);
            end
        else
            if flag == 0
                solved(row, hfun_cases) = 1;
                % solved(row, hfun_cases) = xk_best;
            end
        end

        assert(flag ~= -1, "pounders failed");
        assert(hfun(F(1, :)) > hfun(F(xk_best, :)), "Didn't find decrease over the starting point");
        assert(size(X, 1) <= nf_max + nfs, "POUNDERs grew the size of X");

        if flag == 0
            assert(size(X, 1) <= nf_max + nfs, "POUNDERs evaluated more than nf_max evaluations");
        elseif flag ~= -6 && flag ~= -4 && flag ~= -2
            assert(size(X, 1) == nf_max + nfs, "POUNDERs didn't use nf_max evaluations");
        end

        Results{hfun_cases, row}.alg = 'POUNDERs';
        Results{hfun_cases, row}.problem = ['problem ' num2str(row) ' from More/Wild'];
        Results{hfun_cases, row}.Fvec = F;
        Results{hfun_cases, row}.H = hF;
        Results{hfun_cases, row}.X = X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     save('-mat7-binary', filename, 'Results') % Octave save
        save(filename, 'Results');
    end
end
if ~ensure_still_solve_problems
    writematrix(solved, './benchmark_results/solved.txt');
end

path(oldpath);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
[~, fvec, ~] = calfun(x, struct, probtype);
end
