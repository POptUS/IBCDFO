% Compares the matlab and python implementations of a method.
% This comparison code assumes you've already:
% 1) Run m/tests/benchmark_manifold_sampling.m
% 2) Run py/tests/test_benchmark_manifold_sampling.py
METHOD = 'manifold_sampling';

LW = 2;
FS = 12;
LABEL_FS = 12;

NFMAX = 100;

ROWS = [1, 2, 7, 8, 43, 44, 45];
N_HFUNS = 9; % number of hfuns_all entries in m/tests/benchmark_manifold_sampling.m

filename = ['m/tests/benchmark_results/' METHOD 'M_nf_max=' int2str(NFMAX) '.mat'];
M1 = load(filename);

filename = ['py/tests/msp_benchmark_results/' METHOD '_py_nf_max=' int2str(NFMAX) '.mat'];
P1 = load(filename);
assert(numel(fieldnames(P1)) == N_HFUNS * numel(ROWS));

% MATLAB's hfuns_all and Python's hfuns lists are not sorted identically, so
% pair results up by their hfun_name field rather than by loop position.
np = 0;
for row = ROWS
    M_by_name = dictionary;
    for jj = 1:size(M1.Results, 1)
        Mjj = M1.Results{jj, row};
        M_by_name(Mjj.hfun_name) = Mjj;
    end
    assert(M_by_name.numEntries == N_HFUNS);

    P_by_name = dictionary;
    for i = 0:(N_HFUNS - 1)
        Pi = P1.(['MSP_' int2str(row) '_' int2str(i)]);
        P_by_name(Pi.hfun_name) = Pi;
    end
    assert(isempty(setxor(M_by_name.keys, P_by_name.keys)));

    for name = M_by_name.keys'
        Mk = M_by_name(name);
        Pk = P_by_name(name);

        np = np + 1;
        M{np} = Mk;
        P{np} = Pk;
    end
end

addpath('../../BenDFO/profiling/');

% Data profiles need a common budget across all problems/solvers. We track the
% largest actual evaluation count seen so that truncation, if any, is visible
% to whoever is reading the resulting plots.
max_evals_seen = 0;
prob_dim = zeros(np, 1);
H = inf(NFMAX, np, 2);
Solvers = {[METHOD '-M'], [METHOD '-py']};
for k = 1:np
    max_evals_seen = max(max_evals_seen, length(M{k}.H));
    len = min([NFMAX, length(M{k}.H)]);
    H(1:len, k, 1) = M{k}.H(1:len);

    max_evals_seen = max(max_evals_seen, length(P{k}.H));
    len = min([NFMAX, length(P{k}.H)]);
    H(1:len, k, 2) = P{k}.H(1:len);

    prob_dim(k) = size(M{k}.X, 2);
end

if max_evals_seen > NFMAX
    fprintf(['Note: at least one result used more than nfmax=%d evaluations ' ...
        '(max observed: %d). Data profiles below only count each result''s ' ...
        'first nfmax evaluations.\n'], NFMAX, max_evals_seen);
end

for tau = logspace(-5, -1, 3)
    f = figure;
    h = data_profile(H, prob_dim + 1, tau);

    set(findall(f, 'type', 'line'), 'LineWidth', LW);
    set(gca, 'FontSize', FS);

    legend(h, Solvers, 'Location', 'SouthEast');

    title(sprintf('Results capped at nfmax=%d (max evaluations observed across all results: %d)', NFMAX, max_evals_seen), 'FontSize', FS);
    xlabel('Function evaluations divided by (n+1)', 'FontSize', LABEL_FS);
    ylabel('Fraction of problems solved', 'FontSize', LABEL_FS);

    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];

    print(f, ['Fvalue_data_tau=' num2str(tau) '_nfmax=' int2str(NFMAX) '_' Solvers{1} '_vs_' Solvers{2} '.png'], '-dpng', '-r400');
    close all;
end
