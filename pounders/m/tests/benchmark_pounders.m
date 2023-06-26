% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = benchmark_pounders()

load dfo.dat;

ensure_still_solve_problems = 0;
if ensure_still_solve_problems
    solved = load('./benchmark_results/solved.txt'); % A 0-1 matrix with 1 when problem was previously solved.
else
    solved = zeros(53, 3);
end

spsolver = 2; % TRSP Solver
nfmax = 100;
gtol = 1e-13;
factor = 10;

for row = 1:length(dfo)
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.m = m;
    BenDFO.n = n;

    objective = @(x)calfun_wrapper(x, BenDFO, 'smooth');

    X0 = dfoxs(n, nprob, factor^factor_power)';
    npmax = 2 * n + 1;  % Maximum number of interpolation points [2*n+1]
    L = -Inf(1, n);
    U = Inf(1, n);
    nfs = 0;
    F0 = [];
    xkin = 1;
    delta = 0.1;
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

        filename = ['./benchmark_results/poundersM_nfmax=' int2str(nfmax) '_gtol=' num2str(gtol) '_prob=' int2str(row) '_spsolver=' num2str(spsolver) '_hfun=' func2str(combinemodels) '.mat'];

        [X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);

        if ensure_still_solve_problems
            if solved(row, hfun_cases) == 1
                assert(flag == 0, "This problem was previously solved but it's anymore.");
                check_stationary(X(xk_best, :), L, U, BenDFO, combinemodels);
            end
        else
            if flag == 0
                solved(row, hfun_cases) = xk_best;
            end
        end

        assert(flag ~= -1, "pounders failed");
        assert(hfun(F(1, :)) > hfun(F(xk_best, :)), "Didn't find decrease over the starting point");
        assert(size(X, 1) <= nfmax + nfs, "POUNDERs grew the size of X");

        if flag == 0
            assert(size(X, 1) <= nfmax + nfs, "POUNDERs evaluated more than nfmax evaluations");
        elseif flag ~= -4
            assert(size(X, 1) == nfmax + nfs, "POUNDERs didn't use nfmax evaluations");
        end

        evals = size(F, 1);
        h = zeros(evals, 1);
        for i = 1:evals
            h(i) = hfun(F(i, :));
        end

        Results{hfun_cases, row}.alg = 'POUNDERs';
        Results{hfun_cases, row}.problem = ['problem ' num2str(row) ' from More/Wild'];
        Results{hfun_cases, row}.Fvec = F;
        Results{hfun_cases, row}.H = h;
        Results{hfun_cases, row}.X = X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     save('-mat7-binary', filename, 'Results') % Octave save
        save(filename, 'Results');
    end
end
if ~ensure_still_solve_problems
    writematrix(solved, './benchmark_results/solved.txt');
end
end

function [fvec] = calfun_wrapper(x, struct, probtype)
[~, fvec, ~] = calfun(x, struct, probtype);
end
