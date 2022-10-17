% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = benchmark_pounders()

spsolver = 1;

addpath('../');
addpath('../general_h_funs');
bendfo_location = '../../../../BenDFO';
addpath([bendfo_location, '/m/']);
addpath([bendfo_location, '/data/']);
mkdir('benchmark_results');

nfmax = 100;
gtol = 1e-13;
factor = 10;

load dfo.dat;

for row = 1:53
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
    L = -Inf(1, n);
    U = Inf(1, n);
    nfs = 0;
    X0 = xs';
    F0 = [];
    xkin = 1;
    delta = 0.1;
    printf = 0;

    objective = @(x)calfun_wrapper(x, BenDFO, 'smooth');

    for hfun_cases = 3
        [row, hfun_cases];

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
        end

        filename = ['./benchmark_results/poundersM_nfmax=' num2str(nfmax) '_gtol=' num2str(gtol) '_spsolver=' num2str(spsolver) '_' func2str(combinemodels) '.mat'];
        if exist(filename, 'file')
            Old_results = load(filename);
            re_check = 1;
        else
            Results = cell(1, 53);
            re_check = 0;
        end

        [X, F, flag, xkin] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);

        assert(hfun(F(1, :)) > hfun(F(xkin, :)), "Didn't finds decrease over the starting point");

        assert(size(X, 1) <= nfmax, "POUNDERs grew the size of X");
%         if re_check
%             assert(min(Old_results.Results{1,row}.H) == min(sum(F.^2,2)), "Didn't find the same min")
%             if row~=2 && row~=26 && row~=52
%                 assert(all(all(Old_results.Results{1,row}.Fvec == F)), "Didn't find the same Fvec")
%             end
%         end

        SolverNumber = SolverNumber + 1;
        Results{SolverNumber, row}.alg = 'POUNDERs';
        Results{SolverNumber, row}.problem = ['problem ' num2str(row) ' from More/Wild'];
        Results{SolverNumber, row}.Fvec = F;
        Results{SolverNumber, row}.H = sum(F.^2, 2);
        Results{SolverNumber, row}.X = X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     save('-mat7-binary', filename, 'Results') % Octave save
        save(filename, 'Results');
    end
end
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
end
