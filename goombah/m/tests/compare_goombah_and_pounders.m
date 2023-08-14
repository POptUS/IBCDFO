% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = compare_goombah_and_pounders()
addpath('../../../../BenDFO/data/');
addpath('../../../../BenDFO/m/');
addpath('../../../minq/m/minq5');
addpath('../../../goombah/m');
addpath('../../../pounders/m/general_h_funs');
addpath('../../../pounders/m');
addpath('../../../goombah/m/subproblems/');
addpath('../../../manifold_sampling/m');
addpath('../../../manifold_sampling/m/h_examples/');

load dfo.dat;

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
    printf = 0;

    GAMS_options.file = '../../../goombah/m/subproblems/minimize_sum_quad_models_squared.gms';
    GAMS_options.solvers = 1:4;
    subprob_switch = 'linprog';

    for hfun_cases = 1:1
        Results = cell(2, 3, 53);
        if hfun_cases == 1
            hfun = @(F)sum(F.^2);
            combinemodels = @leastsquares;
        elseif hfun_cases == 2
            error("Not implemented in GOOMBAH yet");
            alpha = 0; % If changed here, also needs to be adjusted in squared_diff_from_mean.m
            hfun = @(F)sum((F - 1 / length(F) * sum(F)).^2) - alpha * (1 / length(F) * sum(F))^2;
            combinemodels = @squared_diff_from_mean;
        elseif hfun_cases == 3
            error("Not implemented in GOOMBAH yet");
            if m ~= 3 % Emittance is only defined for the case when m == 3
                continue
            end
            hfun = @emittance_h;
            combinemodels = @emittance_combine;
        end
        disp([row, hfun_cases]);

        filename = ['./benchmark_results/poundersM_and_GOOMBAH_nfmax=' int2str(nfmax) '_gtol=' num2str(gtol) '_prob=' int2str(row) '_spsolver=' num2str(spsolver) '_hfun=' func2str(combinemodels) '.mat'];

        for method = [1, 2]
            if method == 1
                [X, F, flag, xk_best] = pounders(objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun, combinemodels);
                Results{method, hfun_cases, row}.alg = 'POUNDERs';
            elseif method == 2
                [X, F, h, xkin] = goombah(@sum_squared, objective, nfmax, X0, L, U, GAMS_options, subprob_switch);
                Results{method, hfun_cases, row}.alg = 'GOOMBAH';
            end

            evals = size(F, 1);
            h = zeros(evals, 1);
            for i = 1:evals
                h(i) = hfun(F(i, :));
            end
        end

        Results{method, hfun_cases, row}.problem = ['problem ' num2str(row) ' from More/Wild'];
        Results{method, hfun_cases, row}.Fvec = F;
        Results{method, hfun_cases, row}.H = h;
        Results{method, hfun_cases, row}.X = X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %     save('-mat7-binary', filename, 'Results') % Octave save
        save(filename, 'Results');
    end
end
end

function [fvec] = calfun_wrapper(x, struct, probtype)
[~, fvec, ~] = calfun(x, struct, probtype);
end
