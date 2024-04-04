% Compares the matlab and python implementations of a method.

method = 'manifold_sampling';

LW = 2;
FS = 12;
Label_FS = 12;

nfmax = 1000;

filename = ['m/tests/benchmark_results/' method 'M_nfmax=' int2str(nfmax) '_gentype=3.mat'];
M1 = load(filename);

rows = [1, 2, 7, 8, 43, 44, 45];
cols = [1:7];

countm = 0;
for row = rows
    for col = cols
        countm = countm + 1';
        M{countm} = M1.Results{col, row};
    end
end
filename = ['py/tests/benchmark_results/' method '_py_nfmax=' int2str(nfmax) '.mat'];
P1 = load(filename);

countpy = 0;
for row = rows
    for col = cols
        countpy = countpy + 1';
        P{countpy} = P1.(['MSP_' int2str(row) '_' int2str(col-1)]);
    end
end

assert(all(size(M1) == size(P1)));

np = countm;
ns = 2;
prob_dim = zeros(np, 1);
H = inf(nfmax, np, ns);
Solvers = {[method '-M'], [method '-py']};

addpath('../../BenDFO/profiling/');

for k = 1:np
    for s = 1:ns
        if s == 1
            len = min([nfmax, length(M{k}.H)]);
            H(1:len, k, s) = M{k}.H(1:len);
        elseif s == 2
            len = min([nfmax, length(P{k}.H)]);
            H(1:len, k, s) = P{k}.H(1:len);
        end
    end
    prob_dim(k) = size(M{k}.X, 2);
    %     assert(all(all(M{k}.X(1:n+1,:) == P{k}.X(1:n+1,:))), "The first n+1 points differ between the Matlab and Python versions");
    %     if method == 'pounders'
    %         assert(all(all(M{k}.Fvec(1:n+1,:) == P{k}.Fvec(1:n+1,:))), "The first n+1 Fvec values differ between the Matlab and Python versions");
    %     end
    %     assert(all(all(M{k}.H(1:n+1) == P{k}.H(1:n+1)')), "The first n+1 H values differ between the Matlab and Python versions");
end

for tau = logspace(-5, -1, 3)
    f = figure;
    [h, T] = data_profile(H, prob_dim + 1, tau);

    set(findall(f, 'type', 'line'), 'LineWidth', LW);
    set(gca, 'FontSize', FS);

    legend(h, Solvers, 'Location', 'SouthEast');

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

    print(f, ['Fvalue_data_tau=' num2str(tau) '_nfmax=' int2str(nfmax) '_' Solvers{1} '_vs_' Solvers{2} '.png'], '-dpng', '-r400');
    close all;
end
