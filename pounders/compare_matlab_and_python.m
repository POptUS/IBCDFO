% Compares the matlab and python implementations of a method.

% method = 'orbit'; gtol = 1e-9;
method = 'pounders';

spsolver = 2; % TRSP_SOLVER_MINQ5 in both implementations; see m/create_trsp_solver.m / py/constants.py
LW = 2;
FS = 12;
Label_FS = 12;

nf_max = 100;

% Canonical hfun names, matching the stripped hfun_name used to build the
% filenames written by m/tests/benchmark_pounders.m and
% py/tests/TestPoundersExtensive.py.
hfun_names = {'leastsquares', 'squared_diff_from_mean', 'emittance'};

% Both implementations now write one result per (row, hfun) to a file whose
% name is built identically on both sides (same nf_max/prob/spsolver/hfun_name
% convention), so pairing a MATLAB result with its Python counterpart by that
% shared filename is itself the name-based handshake: two files only get
% compared if their filenames -- and therefore their row and hfun -- agree.
np = 0;
for row = 1:53
    for col = 1:numel(hfun_names)
        hfun_name = hfun_names{col};
        result_name = ['pounders_nf_max=' int2str(nf_max) '_prob=' int2str(row) '_spsolver=' int2str(spsolver) '_hfun=' hfun_name '.mat'];
        filename_m = fullfile('m', 'tests', 'TempPoundersBenchmarkResults', result_name);
        filename_p = fullfile('py', 'tests', 'TempPoundersBenchmarkResults', result_name);
        if isfile(filename_m) && isfile(filename_p)
            Mk = load(filename_m);
            Pk = load(filename_p);

            assert(strcmp(strtrim(Mk.alg), 'POUNDERS_M'), ['Unexpected alg in ' filename_m]);
            assert(strcmp(strtrim(Pk.alg), 'POUNDERS_Py'), ['Unexpected alg in ' filename_p]);

            np = np + 1;
            M{np} = Mk;
            P{np} = Pk;
        end
    end
end

if np == 0
    error('compare_matlab_and_python:noResults', [ ...
        'No paired MATLAB/Python benchmark results were found in\n' ...
        '  %s\nand\n  %s\n' ...
        'Run pounders/m/tests/benchmark_pounders.m and pounders/py/tests/TestPoundersExtensive.py ' ...
        '(from within their own directories) to (re)generate results before comparing.'], ...
        fullfile(pwd, 'm', 'tests', 'TempPoundersBenchmarkResults'), ...
        fullfile(pwd, 'py', 'tests', 'TempPoundersBenchmarkResults'));
end
prob_dim = zeros(np, 1);
H = inf(nf_max, np, 2);
Solvers = {[method '-M'], [method '-py']};

addpath('../../BenDFO/profiling/');

% Data profiles need a common budget across all problems/solvers. We track the
% largest actual evaluation count seen so that truncation, if any, is visible
% to whoever is reading the resulting plots.
max_evals_seen = 0;
for k = 1:np
    max_evals_seen = max(max_evals_seen, length(M{k}.H));
    len = min([nf_max, length(M{k}.H)]);
    H(1:len, k, 1) = M{k}.H(1:len);

    max_evals_seen = max(max_evals_seen, length(P{k}.H));
    len = min([nf_max, length(P{k}.H)]);
    H(1:len, k, 2) = P{k}.H(1:len);

    prob_dim(k) = size(M{k}.X, 2);
end

if max_evals_seen > nf_max
    fprintf(['Note: at least one result used more than nf_max=%d evaluations ' ...
        '(max observed: %d). Data profiles below only count each result''s ' ...
        'first nf_max evaluations.\n'], nf_max, max_evals_seen);
end

for tau = logspace(-7, -1, 7)
    f = figure;
    h = data_profile(H, prob_dim + 1, tau);

    set(findall(f, 'type', 'line'), 'LineWidth', LW);
    set(gca, 'FontSize', FS);

    legend(h, Solvers, 'Location', 'SouthEast');

    title(sprintf('Results capped at nf\\_max=%d (max evaluations observed across all results: %d)', nf_max, max_evals_seen), 'FontSize', FS);
    xlabel('Function evaluations divided by (n+1)', 'FontSize', Label_FS);
    ylabel('Fraction of problems solved', 'FontSize', Label_FS);

    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];

    print(f, ['Fvalue_data_tau=' num2str(tau) '_nf_max=' int2str(nf_max) '_' Solvers{1} '_vs_' Solvers{2} '.png'], '-dpng', '-r400');
    close all;
end

for row = 1:53
    for col = 1:numel(hfun_names)
        hfun_name = hfun_names{col};
        result_name = ['pounders_nf_max=' int2str(nf_max) '_prob=' int2str(row) '_spsolver=' int2str(spsolver) '_hfun=' hfun_name '.mat'];
        filename_m = fullfile('m', 'tests', 'TempPoundersBenchmarkResults', result_name);
        filename_p = fullfile('py', 'tests', 'TempPoundersBenchmarkResults', result_name);
        if isfile(filename_m) && isfile(filename_p)
            Mat = load(filename_m);
            Py = load(filename_p);

            f = figure;
            hold off;
            semilogy(Mat.H, 'LineWidth', LW);
            hold on;
            semilogy(Py.H, 'LineWidth', LW);
            print(f, ['raw_values_row=' int2str(row) '_hfun=' hfun_name '_' Solvers{1} '_vs_' Solvers{2} '.png'], '-dpng', '-r400');
            close all;
        end
    end
end
