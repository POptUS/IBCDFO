% This driver benchmarks algorithms on composite nonsmooth problems of the form
% h(F(x)) where F and the starting point x0 come from the More-Wild SIOPT
% paper "Benchmarking derivative-free optimization algorithms" and h takes
% a variety of forms.
% - F maps from R^n to R^m
% - h maps from R^m to R
% Note that the papers have (p) instead of (m).

global m n nprob probtype vecout % For defintion of F
global C D eqtol % For censored_L1_loss h
global Qs zs cs h_activity_tol % for piecewise_quadratic h
% global mw_prob_num hfun % For saving when there is a model-building failure.

probtype = 'smooth';
vecout = 1;

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

% Declare parameters for benchmark study
nfmax_c = 20; % Full tests used 100; % Multiplied by dimension to set max evals
factor = 10; % Multiple for x0 declaration
num_solvers = 4; % Number of solvers being benchmarked
solver_names = {'MS-D', 'GOOMBAH', 'MS-P', 'GOOMBAH+MS-P'}; % Used when saving filenames for ease of reference

num_seeds = 1; % Replications of each problem instance
mkdir('./benchmark_results');

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
    fprintf("\n**\n  Might need to download this data from: https://web.cels.anl.gov/~jmlarson/mpc_test_files.zip -- Recommend putting in bendfo_root/data/ \n**\n\n");
    Qzb = load('Q_z_and_b_for_benchmark_problems_normalized.mat')';
end
h_activity_tol = 1e-8;

Results = cell(num_solvers, num_seeds, size(dfo, 1));
for mw_prob_num = [7] % full tests are 1:53
    % for constr = [false, true]
    for constr = [false] % Full tests are also true
        nprob = dfo(mw_prob_num, 1);
        n = dfo(mw_prob_num, 2);
        m = dfo(mw_prob_num, 3);
        factor_power = dfo(mw_prob_num, 4);
        x0 = dfoxs(n, nprob, factor^factor_power)';
        nfmax = nfmax_c * (n + 1);

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

                if constr
                    bounds = load(['./test_problems/' func2str(hfun{1}) '_bounds.mat']);
                    LB = bounds.Masterbounds{mw_prob_num}.LB;
                    UB = bounds.Masterbounds{mw_prob_num}.UB;
                else
                    LB = -inf(1, n);
                    UB = inf(1, n);
                end

                for s = 1:4

                    filename = ['./benchmark_results/' solver_names{s} '_prob=' int2str(mw_prob_num) '_seed=' int2str(seed) '_' func2str(hfun{1}) '_nfmax_c=' num2str(nfmax_c) '_constr=' int2str(constr) '.mat'];
                    if exist(filename, 'file')
                        continue
                    end
                    system(['touch ' filename]);

                    if s == 1
                        continue
                        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % % Dual (SIOPT) manifold sampling
                        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % [X, F, h, xkin, flag] = manifold_sampling_SIOPT(hfun{1}, Ffun, x0, nfmax);

                        % assert(size(X, 1) <= nfmax, "Method grew the size of X");

                        % Results{s, seed, mw_prob_num}.alg = solver_names{s};
                        % Results{s, seed, mw_prob_num}.problem = ['problem ' num2str(mw_prob_num) ' from More/Wild with hfun='];
                        % Results{s, seed, mw_prob_num}.Fvec = F;
                        % Results{s, seed, mw_prob_num}.H = h;
                        % Results{s, seed, mw_prob_num}.X = X;
                        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    elseif s == 2
                        if strcmp(func2str(hfun{1}), 'pw_minimum_squared')
                            GAMS_options.file = [trsp_root 'minimize_min_squared_quadratic_models.gms'];
                            GAMS_options.solvers = 1:3;
                        elseif strcmp(func2str(hfun{1}), 'pw_maximum_squared')
                            GAMS_options.file = [trsp_root 'minimize_max_squared_quadratic_models.gms'];
                            GAMS_options.solvers = 1:4;
                        elseif strcmp(func2str(hfun{1}), 'censored_L1_loss')
                            save_censored_L1_loss_data(C, D);
                            GAMS_options.file = [trsp_root 'minimize_censored_L1_loss_quadratic_models.gms'];
                            GAMS_options.solvers = 1:4;
                        elseif strcmp(func2str(hfun{1}), 'piecewise_quadratic')
                            save_piecewise_quadratic_data(Qs, zs, cs);
                            GAMS_options.file = [trsp_root 'minimize_max_quadratic_mapping_of_quadratic_models.gms'];
                            GAMS_options.solvers = 1:4;
                        end

                        [X, F, h, xkin] = goombah_wo_msp(hfun{1}, Ffun, nfmax, x0, LB, UB, GAMS_options);

                        Results{s, seed, mw_prob_num}.alg = solver_names{s};
                        Results{s, seed, mw_prob_num}.problem = ['problem ' num2str(mw_prob_num) ' from More/Wild with hfun='];
                        Results{s, seed, mw_prob_num}.Fvec = F;
                        Results{s, seed, mw_prob_num}.H = h;
                        Results{s, seed, mw_prob_num}.X = X;

                    elseif s == 3
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % PRIMAL MANIFOLD SAMPLING %%%%
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % subprob_switch = 'GAMS_LP'; % subprob_switch = 'GAMS_QCP';
                        subprob_switch = 'linprog';

                        [X, F, h, xkin, flag] = manifold_sampling_primal(hfun{1}, Ffun, x0, LB, UB, nfmax, subprob_switch);

                        assert(size(X, 1) <= nfmax, "Method grew the size of X");

                        Results{s, seed, mw_prob_num}.alg = solver_names{s};
                        Results{s, seed, mw_prob_num}.problem = ['problem ' num2str(mw_prob_num) ' from More/Wild with hfun='];
                        Results{s, seed, mw_prob_num}.Fvec = F;
                        Results{s, seed, mw_prob_num}.H = h;
                        Results{s, seed, mw_prob_num}.X = X;

                    elseif s == 4
                        if strcmp(func2str(hfun{1}), 'pw_minimum_squared')
                            GAMS_options.file = [trsp_root 'minimize_min_squared_quadratic_models.gms'];
                            GAMS_options.solvers = 1:3;
                        elseif strcmp(func2str(hfun{1}), 'pw_maximum_squared')
                            GAMS_options.file = [trsp_root 'minimize_max_squared_quadratic_models.gms'];
                            GAMS_options.solvers = 1:4;
                        elseif strcmp(func2str(hfun{1}), 'censored_L1_loss')
                            save_censored_L1_loss_data(C, D);
                            GAMS_options.file = [trsp_root 'minimize_censored_L1_loss_quadratic_models.gms'];
                            GAMS_options.solvers = 1:4;
                        elseif strcmp(func2str(hfun{1}), 'piecewise_quadratic')
                            save_piecewise_quadratic_data(Qs, zs, cs);
                            GAMS_options.file = [trsp_root 'minimize_max_quadratic_mapping_of_quadratic_models.gms'];
                            GAMS_options.solvers = 1:4;
                        end
                        % subprob_switch = 'GAMS_LP'; % subprob_switch = 'GAMS_QCP';
                        subprob_switch = 'linprog';

                        [X, F, h, xkin] = goombah(hfun{1}, Ffun, nfmax, x0, LB, UB, GAMS_options, 'GAMS_LP');

                        Results{s, seed, mw_prob_num}.alg = solver_names{s};
                        Results{s, seed, mw_prob_num}.problem = ['problem ' num2str(mw_prob_num) ' from More/Wild with hfun='];
                        Results{s, seed, mw_prob_num}.Fvec = F;
                        Results{s, seed, mw_prob_num}.H = h;
                        Results{s, seed, mw_prob_num}.X = X;

                    end % switch on s

                    save(filename, 'Results');
                end
            end
        end
    end
end
