% This compares goombah and goombah_wo_msp, using the LHS TRSP solver instead of
% GAMS, against manifold_sampling_primal on a subset of the composite nonsmooth
% problems h(F(x)) from the More and Wild SIOPT paper "Benchmarking derivative-free
% optimization algorithms". The problems and data are those of
% benchmark_manifold_sampling.m.
%
% For each problem, we assert that each GOOMBAH variant decreases h at least 90%
% as much as manifold_sampling_primal does from the same starting point, and that
% the outputs are consistent (see check_outputs).
%
% To run this test, you must first install a BenDFO clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
% to the MATLAB path.

function [] = test_goombah_vs_msp()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'goombah', 'm', 'subproblems'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));
addpath(fullfile(repo_root, 'pounders', 'm'));

factor = 10;
nf_max = 100;
subprob_switch = 'linprog';
TRSP_options.solver = 'lhs';

load dfo.dat;

if ~exist("mpc_test_files_smaller_Q", "dir")
    url = "https://web.cels.anl.gov/~jmlarson/mpc_test_files_smaller_Q.zip";
    if system("which wget > /dev/null") == 0
        system("wget " + url);
    else
        % MATLAB puts its own bundled libcurl on the loader path, which
        % conflicts with the system curl binary unless cleared here.
        system("LD_LIBRARY_PATH='' DYLD_LIBRARY_PATH='' curl -O -L " + url);
    end
    system("unzip mpc_test_files_smaller_Q.zip");
end

C_L1_loss = load('mpc_test_files_smaller_Q/C_for_benchmark_probs.csv');
D_L1_loss = load('mpc_test_files_smaller_Q/D_for_benchmark_probs.csv');
Qzb = load('mpc_test_files_smaller_Q/Q_z_and_b_for_benchmark_problems_normalized_subset.mat');

hfun_names = {'create_censored_L1_loss_hfun', ...
              'create_piecewise_quadratic_hfun', ...
              'h_max_plus_quadratic_violation_penalty', ...
              'h_pw_maximum', 'h_pw_maximum_squared', ...
              'h_pw_minimum', 'h_pw_minimum_squared', ...
              'h_quantile', ...
              'h_one_norm'};

% The slow cases (censored_L1_loss and one_norm on row 43) are omitted to keep
% the runtime reasonable.
rows_and_hfuns = {1, [5, 7]; ...
                  7, 1:9; ...
                  43, [2, 3, 4, 5, 7, 8]};

for i = 1:size(rows_and_hfuns, 1)
    row = rows_and_hfuns{i, 1};
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    LB = -5000 * ones(1, n);
    UB = 5000 * ones(1, n);

    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    x0 = dfoxs(n, nprob, factor^factor_power)';

    ind = find(C_L1_loss(:, 1) == row & C_L1_loss(:, 2) == 1);
    C = C_L1_loss(ind, 4:m + 3);
    D = D_L1_loss(ind, 4:m + 3);

    hfuns_all = {create_censored_L1_loss_hfun(C, D), ...
                 create_piecewise_quadratic_hfun(Qzb.Q_mat{row, 1}, Qzb.z_mat{row, 1}, Qzb.b_mat{row, 1}), ...
                 @h_max_plus_quadratic_violation_penalty, ...
                 @h_pw_maximum, @h_pw_maximum_squared, ...
                 @h_pw_minimum, @h_pw_minimum_squared, ...
                 @h_quantile, ...
                 @h_one_norm};

    for jj = rows_and_hfuns{i, 2}
        hfun = hfuns_all{jj};
        problem = ['problem ' num2str(row) ' with hfun=' hfun_names{jj}];

        [~, ~, h_msp, ~, ~] = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nf_max, subprob_switch);
        decrease_msp = h_msp(1) - min(h_msp);
        assert(decrease_msp > 0, "manifold_sampling_primal did not decrease h on " + problem);

        rand('seed', 1000 * row + jj);
        [X, F, h, xkin] = goombah(hfun, Ffun, nf_max, x0, LB, UB, TRSP_options, subprob_switch);
        check_outputs(X, F, h, xkin);
        assert(size(X, 1) <= nf_max, "goombah grew the size of X on " + problem);
        assert(h(1) - min(h) >= 0.9 * decrease_msp, "goombah decreased h much less than manifold_sampling_primal on " + problem);

        rand('seed', 1000 * row + jj);
        [X, F, h, xkin] = goombah_wo_msp(hfun, Ffun, nf_max, x0, LB, UB, TRSP_options);
        check_outputs(X, F, h, xkin);
        assert(size(X, 1) <= nf_max, "goombah_wo_msp grew the size of X on " + problem);
        assert(h(1) - min(h) >= 0.9 * decrease_msp, "goombah_wo_msp decreased h much less than manifold_sampling_primal on " + problem);
    end
end

path(oldpath);
end

function [] = check_outputs(X, F, h, xkin)
    assert(size(X, 1) == size(F, 1) && size(F, 1) == numel(h), "Inconsistent output sizes");
    assert(h(xkin) < h(1), "No decrease from the starting point");
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
