% This wrapper tests various algorithms against the Benchmark functions from the
% More and Wild SIOPT paper "Benchmarking derivative-free optimization algorithms"
function [] = benchmark_manifold_sampling()

global C D Qs zs cs
nfmax = 100;
factor = 10;

subprob_switch = 'linprog';

load dfo.dat;

filename = ['./benchmark_results/manifold_samplingM_nfmax=' num2str(nfmax) '.mat'];
Results = cell(1, 53);

if ~exist("mpc_test_files_smaller_Q", "dir")
    system("wget https://web.cels.anl.gov/~jmlarson/mpc_test_files_smaller_Q.zip");
    system("unzip mpc_test_files_smaller_Q.zip");
end

C_L1_loss = load('mpc_test_files_smaller_Q/C_for_benchmark_probs.csv');
D_L1_loss = load('mpc_test_files_smaller_Q/D_for_benchmark_probs.csv');
Qzb = load('mpc_test_files_smaller_Q/Q_z_and_b_for_benchmark_problems_normalized_subset.mat')';

% for row = find(cellfun(@length,Results)==0)
for row = [2, 1, 7, 8, 43, 44, 45]
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

    ind = find(C_L1_loss(:, 1) == row & C_L1_loss(:, 2) == 1);
    C = C_L1_loss(ind, 4:m + 3);
    D = D_L1_loss(ind, 4:m + 3);
    Qs = Qzb.Q_mat{row, 1};
    zs = Qzb.z_mat{row, 1};
    cs = Qzb.b_mat{row, 1};
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Manifold sampling
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    jj = 1;
    for hfuns = {@censored_L1_loss_quad_MSG, @censored_L1_loss, @max_sum_beta_plus_const_viol, @piecewise_quadratic, @piecewise_quadratic_1, @pw_maximum,  @pw_maximum_squared, @pw_minimum, @pw_minimum_squared, @quantile}
        hfun = hfuns{1};
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
