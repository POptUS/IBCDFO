% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = benchmark_manifold_sampling()

nfmax = 50;
factor = 10;

subprob_switch = 'linprog';

load dfo.dat;

filename = ['./benchmark_results/manifold_samplingM_nfmax=' num2str(nfmax) '.mat'];
Results = cell(1, 53);

C_L1_loss = load('C_for_benchmark_probs.csv');
D_L1_loss = load('D_for_benchmark_probs.csv');
Qzb = load('../mpc_test_files/Q_z_and_b_for_benchmark_problems_normalized.mat')';

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

    ind = find(C_L1_loss(:, 1) == mw_prob_num & C_L1_loss(:, 2) == seed);
    C = C_L1_loss(ind, 4:m + 3);
    D = D_L1_loss(ind, 4:m + 3);
    Qs = Qzb.Q_mat{row, 0};
    zs = Qzb.z_mat{row, 0};
    cs = Qzb.b_mat{row, 0};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Manifold sampling
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    jj = 1;
    for hfuns = {@censored_L1_loss,  @censored_L1_loss_quad_MSG,  @max_sum_beta_plus_const_viol, @piecewise_quadratic,  @pw_maximum,  @pw_maximum_squared,  @pw_minimum, @pw_minimum_squared, @quantile}
        hfun = hfuns{1}
        Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
        x0 = xs';

        [X, F, h, xkin, flag] = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch);

        Results{jj, row}.alg = 'Manifold sampling';
        Results{jj, row}.problem = ['problem ' num2str(row) ' from More/Wild with hfun='];
        Results{jj, row}.Fvec = F;
        Results{jj, row}.H = h;
        Results{jj, row}.X = X;
        jj = jj + 1;
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
