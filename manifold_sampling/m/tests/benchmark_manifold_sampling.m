% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = benchmark_manifold_sampling()

addpath('../');
addpath('../../../../BenDFO/m/');
addpath('../../../../BenDFO/data/');
addpath('../h_examples/');
addpath('../../../pounders/m'); % formquad, bmpts, boxline, phi2eval

mkdir('benchmark_results');

nfmax = 500;
factor = 10;

subprob_switch = 'linprog';

load dfo.dat;

filename = ['./benchmark_results/manifold_samplingM_nfmax=' num2str(nfmax) '.mat'];
Results = cell(1, 53);

% for row = find(cellfun(@length,Results)==0)
for row = [1, 2, 7, 8, 43, 44, 45]
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    LB = -Inf * ones(1, n);
    UB = Inf * ones(1, n);

    xs = dfoxs(n, nprob, factor^factor_power);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Manifold sampling
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for jj = 1:2
        if jj == 1
            hfun = @pw_maximum_squared;
        end
        if jj == 2
            hfun = @pw_minimum_squared;
        end

        Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
        x0 = xs';

        [X, F, h, xkin, flag] = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch);

        Results{jj, row}.alg = 'Manifold sampling';
        Results{jj, row}.problem = ['problem ' num2str(row) ' from More/Wild with hfun='];
        Results{jj, row}.Fvec = F;
        Results{jj, row}.H = h;
        Results{jj, row}.X = X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % save(filename, 'Results');
        % save('-mat7-binary', filename, 'Results') % Octave save
    end
end
save(filename, 'Results');
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
