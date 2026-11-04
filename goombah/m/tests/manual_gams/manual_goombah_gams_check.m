% MANUAL, GAMS-based check of the GOOMBAH GAMS trust-region subproblem flow.
%
% This is NOT part of the automatic test suite (Testgoombah, CI): GAMS is
% licensed software that is unavailable on GitHub runners (Issue 332). Its
% filename deliberately does not begin or end with "test" so that runtests
% does not collect it. Reviewers with a GAMS installation should run it by hand
% when a PR changes GOOMBAH or its GAMS subproblems.
%
% Requirements:
%    * the gams executable on the system path
%    * GDXMRW (wgdx/rgdx) on the MATLAB path. Recent GAMS releases may not ship
%      the MATLAB MEX files; if yours does not, add the GDXMRW folder of an older
%      GAMS installation (the MEX files from an older release worked with a newer
%      gams executable).
%    * BenDFO (https://github.com/POptUS/BenDFO: its data and m folders) and
%      MINQ (https://github.com/POptUS/minq: m/minq5, at the commit in
%      REQUIRED_MINQ_COMMIT in the repository root) on the MATLAB path
%    * internet access (or an unzipped mpc_test_files_smaller_Q in the current
%      folder) for the data used by the parameterized h functions
%
% From MATLAB, in a folder you are willing to have data downloaded into (the first
% run downloads and unzips mpc_test_files_smaller_Q.zip there):
%     >> addpath('/path/to/IBCDFO/goombah/m/tests/manual_gams')
%     >> manual_goombah_gams_check          % default: subproblem comparison
%     >> manual_goombah_gams_check(true)    % also run one short goombah run
%     >> manual_goombah_gams_check(false, true)  % also the slow piecewise case
%
% The default run takes a few minutes. It prints the GAMS and lhs predicted
% decreases for each h function and ends with "All manual GAMS checks passed."; an
% error is raised on the first failure. Reviewers may wish to note the result in
% the PR (no results are reported back automatically).
%
% This only checks that the GAMS flow works (data written to GDX, GAMS called,
% solution read back) and that it is close to the lhs solver. It is deliberately
% small: only the first problem of dfo.dat is used, and no GOOMBAH run is made
% unless asked for.
%
% Default: for each h function, one TRSP is built from fixed random quadratic models
% and solved with both the GAMS and lhs TRSP solvers. The GAMS predicted decrease
% must be at least 90% of the lhs one (only a positive decrease is required for the
% nonconvex min-type h functions), and the GAMS step must be feasible and
% consistent with its reported predicted decrease.
%
% With include_goombah = true: one short goombah run (nf_max = 30) per solver on
% the h_pw_maximum_squared function; the GAMS run must decrease h at least 90% as much as
% the lhs run does.
%
% The create_piecewise_quadratic_hfun case is very slow with GAMS, so it is only
% run with include_slow = true.

function [] = manual_goombah_gams_check(include_goombah, include_slow)

if nargin < 1
    include_goombah = false;
end
if nargin < 2
    include_slow = false;
end

assert(system('which gams > /dev/null 2>&1') == 0, ...
       'The gams executable is not on the system path');
assert(exist('wgdx', 'file') > 0, 'GDXMRW (wgdx) is not on the MATLAB path');

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));
trsp_root = fullfile(repo_root, 'goombah', 'm', 'subproblems');
addpath(trsp_root);

factor = 10;
nf_max = 30;
subprob_switch = 'linprog';
lhs_options = struct('solver', 'lhs');
oldwarn = warning('off', 'GOOMBAH:lhsTrspSolver');

load dfo.dat;

if ~exist('mpc_test_files_smaller_Q', 'dir')
    url = "https://web.cels.anl.gov/~jmlarson/mpc_test_files_smaller_Q.zip";
    if system("which wget > /dev/null") == 0
        system("wget " + url);
    else
        system("LD_LIBRARY_PATH='' DYLD_LIBRARY_PATH='' curl -O -L " + url);
    end
    system("unzip mpc_test_files_smaller_Q.zip");
end
C_L1_loss = load('mpc_test_files_smaller_Q/C_for_benchmark_probs.csv');
D_L1_loss = load('mpc_test_files_smaller_Q/D_for_benchmark_probs.csv');
Qzb = load('mpc_test_files_smaller_Q/Q_z_and_b_for_benchmark_problems_normalized_subset.mat');

% Rows of dfo.dat to test, and the h functions that have a GAMS model file.
rows = 1;
names = {'h_pw_minimum', 'h_pw_minimum_squared', 'h_pw_maximum_squared', ...
         'create_censored_L1_loss_hfun'};
if include_slow
    names{end + 1} = 'create_piecewise_quadratic_hfun';
end

for row = rows
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
    x0_prob = dfoxs(n, nprob, factor^factor_power)';

    ind = find(C_L1_loss(:, 1) == row & C_L1_loss(:, 2) == 1);
    C = C_L1_loss(ind, 4:m + 3);
    D = D_L1_loss(ind, 4:m + 3);
    Q = Qzb.Q_mat{row, 1};
    z = Qzb.z_mat{row, 1};
    c = Qzb.b_mat{row, 1};

    for name = names
        name = name{1}; %#ok<FXSET>
        TRSP_options = struct('solver', 'GAMS', 'hfun_data', []);
        switch name
            case 'h_pw_minimum'
                hfun = @h_pw_minimum;
                TRSP_options.file = fullfile(trsp_root, 'minimize_min_quadratic_models.gms');
                TRSP_options.solvers = 1:3;
            case 'h_pw_minimum_squared'
                hfun = @h_pw_minimum_squared;
                TRSP_options.file = fullfile(trsp_root, 'minimize_min_squared_quadratic_models.gms');
                TRSP_options.solvers = 1:3;
            case 'h_pw_maximum_squared'
                hfun = @h_pw_maximum_squared;
                TRSP_options.file = fullfile(trsp_root, 'minimize_max_squared_quadratic_models.gms');
                TRSP_options.solvers = 1:4;
            case 'create_censored_L1_loss_hfun'
                hfun = create_censored_L1_loss_hfun(C, D);
                TRSP_options.hfun_data = struct('C', C, 'D', D);
                TRSP_options.file = fullfile(trsp_root, 'minimize_censored_L1_loss_quadratic_models.gms');
                TRSP_options.solvers = 1:4;
            case 'create_piecewise_quadratic_hfun'
                hfun = create_piecewise_quadratic_hfun(Q, z, c);
                TRSP_options.hfun_data = struct('Q', Q, 'z', z, 'c', c);
                TRSP_options.file = fullfile(trsp_root, 'minimize_max_quadratic_mapping_of_quadratic_models.gms');
                TRSP_options.solvers = 1:4;
        end
        problem = ['problem ' num2str(row) ' with hfun=' name];
        fprintf('Running %s\n', problem);

        % One TRSP from fixed random quadratic models, GAMS versus lhs.
        rand('seed', 1);
        randn('seed', 1);
        % convex quadratics, so the comparison does not depend on which local
        % minimizer each solver happens to find
        H = zeros(n, n, m);
        for i = 1:m
            A = 0.3 * randn(n, n);
            H(:, :, i) = A' * A;
        end
        g = randn(n, m);
        b = abs(randn(1, m));
        delta = 1;
        Low = -delta * ones(1, n);
        Upp = delta * ones(1, n);
        x0 = zeros(1, n);
        val_at_x0 = hfun(b);

        rand('seed', 1);
        [~, pred_lhs] = minimize_quadratics_lhs(H, g, b, Low, Upp, x0, x0, val_at_x0, hfun);
        [s_gams, pred_gams] = save_quadratics_call_GAMS(H, g, b, Low, Upp, x0, x0, val_at_x0, TRSP_options, hfun);
        fprintf('  predicted decrease: GAMS %g, lhs %g\n', pred_gams, pred_lhs);

        assert(all(size(s_gams) == [1, n]), "GAMS step has wrong size on " + problem);
        assert(all(s_gams >= Low - 1e-6) && all(s_gams <= Upp + 1e-6), "GAMS step infeasible on " + problem);
        z = zeros(1, m);
        for i = 1:m
            z(i) = b(i) + g(:, i)' * s_gams(:) + 0.5 * s_gams(:)' * H(:, :, i) * s_gams(:);
        end
        assert(abs((val_at_x0 - hfun(z)) - pred_gams) <= 1e-6 * max(1, abs(val_at_x0)), ...
               "GAMS predicted decrease inconsistent with its step on " + problem);
        % The min-type TRSPs are nonconvex across pieces: the GAMS NLP solvers
        % find a local solution from x0 while lhs samples globally, so GAMS can
        % legitimately be much worse. Only require a decrease there.
        if any(strcmp(name, {'h_pw_minimum', 'h_pw_minimum_squared'}))
            assert(pred_gams > 0, "GAMS predicted no decrease on " + problem);
        else
            assert(pred_gams >= 0.9 * pred_lhs, "GAMS predicted decrease much less than lhs on " + problem);
        end

        if include_goombah && strcmp(name, 'h_pw_maximum_squared')
            rand('seed', 1);
            [~, ~, h_lhs, ~] = goombah(hfun, Ffun, nf_max, x0_prob, LB, UB, lhs_options, subprob_switch);
            decrease_lhs = h_lhs(1) - min(h_lhs);
            assert(decrease_lhs > 0, "goombah with lhs did not decrease h on " + problem);

            rand('seed', 1);
            [X, F, h, xkin] = goombah(hfun, Ffun, nf_max, x0_prob, LB, UB, TRSP_options, subprob_switch);
            check_outputs(X, F, h, xkin, nf_max, "goombah on " + problem);
            assert(h(1) - min(h) >= 0.9 * decrease_lhs, "goombah GAMS decreased h much less than lhs on " + problem);
        end
    end
end

fprintf('All manual GAMS checks passed.\n');
warning(oldwarn);
path(oldpath);
end

function [] = check_outputs(X, F, h, xkin, nf_max, label)
    assert(size(X, 1) == size(F, 1) && size(F, 1) == numel(h), "Inconsistent output sizes: " + label);
    assert(size(X, 1) <= nf_max, "Exceeded evaluation budget: " + label);
    assert(h(xkin) < h(1), "No decrease from the starting point: " + label);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end
