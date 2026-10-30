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
% To run this test, you must first install a BenDFO and a MINQ clone and add
%    /path/to/BenDFO/data
%    /path/to/BenDFO/m
%    /path/to/MINQ/m/minq5
% to the MATLAB path.

function [] = test_goombah_vs_msp()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));

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

% The slow cases (censored_L1_loss and one_norm on row 43) are omitted to keep
% the runtime reasonable. Empty means all hfuns.
rows_and_hfuns = {1, {'h_pw_maximum_squared', 'h_pw_minimum_squared'}; ...
                  7, {}; ...
                  43, {'create_piecewise_quadratic_hfun', ...
                       'h_max_plus_quadratic_violation_penalty', ...
                       'h_pw_maximum', 'h_pw_maximum_squared', ...
                       'h_pw_minimum_squared', 'h_quantile'}};

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

    % Each row is {name, handle}. The first two hfuns depend on the problem data.
    hfuns = {'create_censored_L1_loss_hfun', create_censored_L1_loss_hfun(C, D); ...
             'create_piecewise_quadratic_hfun', create_piecewise_quadratic_hfun(Qzb.Q_mat{row, 1}, Qzb.z_mat{row, 1}, Qzb.b_mat{row, 1}); ...
             'h_max_plus_quadratic_violation_penalty', @h_max_plus_quadratic_violation_penalty; ...
             'h_pw_maximum', @h_pw_maximum; ...
             'h_pw_maximum_squared', @h_pw_maximum_squared; ...
             'h_pw_minimum', @h_pw_minimum; ...
             'h_pw_minimum_squared', @h_pw_minimum_squared; ...
             'h_quantile', @h_quantile; ...
             'h_one_norm', @h_one_norm};

    selected = rows_and_hfuns{i, 2};
    if isempty(selected)
        selected = hfuns(:, 1)';
    end
    assert(all(ismember(selected, hfuns(:, 1))), "Unknown hfun name in rows_and_hfuns");

    for jj = find(ismember(hfuns(:, 1), selected))'
        hfun = hfuns{jj, 2};
        problem = ['problem ' num2str(row) ' with hfun=' hfuns{jj, 1}];

        [~, ~, h_msp, ~, ~] = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nf_max, subprob_switch);
        decrease_msp = h_msp(1) - min(h_msp);
        assert(decrease_msp > 0, "manifold_sampling_primal did not decrease h on " + problem);

        rand('seed', 1);
        [X, F, h, xkin] = goombah(hfun, Ffun, nf_max, x0, LB, UB, TRSP_options, subprob_switch);
        check_outputs(X, F, h, xkin);
        assert(size(X, 1) <= nf_max, "goombah grew the size of X on " + problem);
        assert(h(1) - min(h) >= 0.9 * decrease_msp, "goombah decreased h much less than manifold_sampling_primal on " + problem);

        rand('seed', 1);
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
