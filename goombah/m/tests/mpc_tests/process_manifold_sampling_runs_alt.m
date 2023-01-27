% This driver processes the results from the benchmarked algorithms in call_many
% call_many_manifold_sampling_methods_on_many_hfuns.m
% Run that file to get the correct paths.

root_dir = '../../../../';
bendfo_root = '../../../../../BenDFO/';
trsp_root = [root_dir 'subproblems/'];

% Add a bunch of paths
addpath([root_dir 'goombah/m/']);
addpath(trsp_root);
addpath([trsp_root 'gqt/']);
addpath([root_dir 'manifold_sampling/m/']);
addpath([root_dir 'manifold_sampling/m/h_examples/']);
addpath([bendfo_root 'data/']);
addpath([bendfo_root 'm/']);
addpath('test_problems/');
addpath([root_dir 'pounders/m/']); % formquad, bmpts, boxline, phi2eval

global m n nprob probtype vecout % For defintion of F
global C D eqtol % For censored_L1_loss h
global Qs zs cs h_activity_tol % for piecewise_quadratic h

probtype = 'smooth';
vecout = 1;

% Declare parameters for benchmark study
nfmax_c = 20; % Multiplied by dimension to set max evals
factor = 10; % Multiple for x0 declaration
num_solvers = 4; % Number of solvers being benchmarked
solver_names = {'MS-D', 'GOOMBAH', 'MS-P', 'GOOMBAH+MS-P'}; % Used when saving filenames for ease of reference
num_seeds = 1; % Replications of each problem instance
mkdir('processed_results');

% Define More-Wild F functions
% Ffun = @calfun;
Ffun = @blackbox;
load dfo.dat;

% Defines data for censored-L1 loss h instances
C_L1_loss = load('C_for_benchmark_probs.csv');
D_L1_loss = load('D_for_benchmark_probs.csv');
eqtol = 1e-8;

% Defines data for piecewise_quadratic h instances
if ~exist('Qzb', 'var')
    Qzb = load('~/public_html/Q_z_and_b_for_benchmark_problems_normalized.mat')';
    % Qzb = load('~/nsdfo20/code/obj_funcs/Q_z_and_b_for_benchmark_problems_normalized.mat')';
end
h_activity_tol = 1e-8;

eps_ball = 1e-5;
num_sample_pts = 50;
for mw_prob_num = [7] % 1:53 % The More-Wild benchmark problem number
    for constr = [false] % [false, true]
        nprob = dfo(mw_prob_num, 1);
        n = dfo(mw_prob_num, 2);
        m = dfo(mw_prob_num, 3);
        factor_power = dfo(mw_prob_num, 4);
        x0 = dfoxs(n, nprob, factor^factor_power)';
        nfmax = nfmax_c * (n + 1);
        eye_n = eye(n);

        for seed = 1:num_seeds
            % Individual for censored-L1 loss h instance
            ind = find(C_L1_loss(:, 1) == mw_prob_num & C_L1_loss(:, 2) == seed);
            C = C_L1_loss(ind, 4:m + 3);
            D = D_L1_loss(ind, 4:m + 3);

            % Individual for piecewise_quadratic h instance
            Qs = Qzb.Q_mat{mw_prob_num, seed};
            zs = Qzb.z_mat{mw_prob_num, seed};
            cs = Qzb.b_mat{mw_prob_num, seed};

            for hfun = {@pw_minimum_squared, @pw_maximum_squared, @censored_L1_loss, @piecewise_quadratic}
                Hist_norm = inf(nfmax, num_solvers);
                Hist_h = inf(nfmax, num_solvers);

                for s = 1:num_solvers

                    run_filename = ['benchmark_results/' solver_names{s} '_prob=' int2str(mw_prob_num) '_seed=' int2str(seed) '_' func2str(hfun{1}) '_nfmax_c=' num2str(nfmax_c) '_constr=' int2str(constr) '.mat'];
                    if ~exist(run_filename, 'file') || dir(run_filename).bytes == 0
                        continue
                    end

                    processed_filename = ['processed_results/processed_' solver_names{s} '_prob=' int2str(mw_prob_num) '_seed=' ...
                                          int2str(seed) '_' func2str(hfun{1}) '_nfmax_c=' num2str(nfmax_c) '_constr=' int2str(constr) '_alt.mat'];
                    if exist(processed_filename, 'file')
                        continue
                    end
                    system(['touch ' processed_filename]);

                    A = load(run_filename);
                    A = A.Results{s, seed, mw_prob_num};

                    % First find the grad at all evaluated points
                    evals = min(nfmax, size(A.Fvec, 1));
                    point_grads = cell(evals, 1);
                    point_fs = cell(evals, 1);
                    for i = 1:evals
                        [fvec, grad_fvec] = Ffun(A.X(i, :));
                        [hout1, hout2, hout3] = hfun{1}(fvec);
                        hout4 = hfun{1}(fvec, hout3);
                        Hist_h(i, s) = hout1;
                        grad_h = hout2;
                        G_k = grad_fvec' * grad_h;
                        point_fs{i} = hout4;
                        point_grads{i} = G_k;
                    end
                    Dists = distmat(A.X(1:evals, :)); % Can get this here: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/15145/versions/3/previews/distmat.m/index.html

                    if constr
                        bounds = load(['../../regression_tests/test_problems/' func2str(hfun{1}) '_bounds.mat']);
                        LB = bounds.Masterbounds{mw_prob_num}.LB;
                        UB = bounds.Masterbounds{mw_prob_num}.UB;
                    else
                        LB = -Inf * ones(1, n);
                        UB = Inf * ones(1, n);
                    end

                    % Then sample num_sample_pts around each evaluated point,
                    % including other (nearby) evaluated points from the run
                    rand('state', seed);
                    randn('state', seed);
                    for i = 1:evals
                        disp(i);

                        close_inds = find(Dists(:, i) <= eps_ball);
                        sampled_grads = [point_grads{close_inds}];
                        f_bar = [point_fs{close_inds}];

                        % CLEAN UP
                        to_remove = any(isnan(sampled_grads));
                        sampled_grads(:, to_remove) = [];
                        f_bar(to_remove) = [];

                        X = randn(num_sample_pts, n);
                        s2 = sum(X.^2, 2);
                        %                 X = X.*repmat(eps_ball*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n); %  On the ball
                        dists = eps_ball * rand(num_sample_pts, 1);
                        X = X .* repmat(dists ./ sqrt(s2), 1, n); %  In the ball

                        for j = 1:num_sample_pts
                            [fvec, grad_fvec] = Ffun(A.X(i, :) + X(j, :));
                            [~, grad_h, hash] = hfun{1}(fvec);
                            hf = hfun{1}(fvec, hash);
                            G_k = grad_fvec' * grad_h;

                            keepers = ~any(isnan(G_k));
                            sampled_grads = [sampled_grads G_k(:, keepers)];
                            f_bar = [f_bar hf(keepers)];
                        end
                        beta = max(0, f_bar' - Hist_h(i, s));
                        subprob_switch = 'GAMS_LP';
                        Low = max(LB - A.X(i, :), -1.0);
                        Upp = min(UB - A.X(i, :), 1.0);
                        [~, ~, chi, ~] = minimize_affine_envelope(Hist_h(i, s), f_bar, beta, sampled_grads, zeros(n), 1.0, Low, Upp, zeros(size(sampled_grads, 2), n + 1, n + 1), subprob_switch);

                        assert(chi >= -1e-12, "This stationary metric should not be this negative.");

                        Hist_norm(i, s) = chi;
                        % if Hist_norm(i, s) == 0 || Hist_norm(i, s) / Hist_norm(1, s) <= 1e-7
                        if Hist_norm(i, s) <= 1e-7
                            break % Stop if the stationary measure is small.
                        end
                    end

                    disp(min(Hist_norm));

                    Solvers = A.alg;
                    save(processed_filename, 'Hist_h', 'Hist_norm', 'Solvers');
                end
            end
        end
    end
end
