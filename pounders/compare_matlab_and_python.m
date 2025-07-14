% Compares the matlab and python implementations of a method.

% method = 'orbit'; gtol = 1e-9;
method = 'pounders';
gtol = 1e-13;

spsolver = 2;
LW = 2;
FS = 12;
Label_FS = 12;

nf_max = 100;
probtype = 'smooth';

countm = 0;
for row = 1:53
    for hfun = {'leastsquares', 'squared_diff_from_mean', 'emittance_combine'}
        filename = ['m/tests/benchmark_results/poundersM_nf_max=' int2str(nf_max) '_gtol=' num2str(gtol) '_prob=' int2str(row) '_spsolver=' num2str(spsolver) '_hfun=' hfun{1} '.mat'];
        if exist(filename)
            if strcmp(hfun, 'leastsquares')
                col = 1;
            elseif strcmp(hfun, 'squared_diff_from_mean')
                col = 2;
            elseif strcmp(hfun, 'emittance_combine')
                col = 3;
            end
            M1 = load(filename);
            countm = countm + 1';
            M{countm} = M1.Results{col, row};
        end
    end
end

countpy = 0;
for row = 1:53
    for hfun = {'leastsquares', 'squared_diff_from_mean', 'emittance_combine'}
        filename = ['py/tests/regression_tests/benchmark_results/' method '4py_nf_max=' int2str(nf_max) '_gtol=' num2str(gtol) '_prob=' int2str(row - 1) '_spsolver=' num2str(spsolver) '_hfun=' hfun{1} '.mat'];
        if exist(filename)
            P1 = load(filename);
            countpy = countpy + 1';
            if strcmp(hfun, 'leastsquares')
                col = 1;
            elseif strcmp(hfun, 'squared_diff_from_mean')
                col = 2;
            elseif strcmp(hfun, 'emittance_combine')
                col = 3;
            end
            P{countpy} = P1.([method '4py_' int2str(row - 1) '_' int2str(col)]);
        end
    end
end
assert(countpy == countm);

np = countm;
ns = 2;
nf = nf_max;
prob_dim = zeros(np, 1);
H = inf(nf_max, np, ns);
Solvers = {[method '-M'], [method '-py']};

addpath('../../BenDFO/profiling/');

for k = 1:np
    for s = 1:ns
        if s == 1
            len = min([nf_max, length(M{k}.H)]);
            H(1:len, k, s) = M{k}.H(1:len);
        elseif s == 2
            len = min([nf_max, length(P{k}.H)]);
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

for tau = logspace(-7, -1, 7)
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

    print(f, ['Fvalue_data_tau=' num2str(tau) '_nf_max=' int2str(len) '_' Solvers{1} '_vs_' Solvers{2} '.png'], '-dpng', '-r400');
    close all;
end

for row = 1:53
    for hfun = {'leastsquares', 'squared_diff_from_mean', 'emittance_combine'}
        filename_m = ['m/tests/benchmark_results/poundersM_nf_max=' int2str(nf_max) '_gtol=' num2str(gtol) '_prob=' int2str(row) '_spsolver=' num2str(spsolver) '_hfun=' hfun{1} '.mat'];
        filename_p = ['py/tests/regression_tests/benchmark_results/' method '4py_nf_max=' int2str(nf_max) '_gtol=' num2str(gtol) '_prob=' int2str(row - 1) '_spsolver=' num2str(spsolver) '_hfun=' hfun{1} '.mat'];
        if exist(filename_p)
            M1 = load(filename_m);
            P1 = load(filename_p);
            col = 0;
            if strcmp(hfun, 'leastsquares')
                col = 1;
            elseif strcmp(hfun, 'squared_diff_from_mean')
                col = 2;
            elseif strcmp(hfun, 'emittance_combine')
                col = 3;
            end
            f = figure;
            Mat = M1.Results{col, row};
            Py = P1.([method '4py_' int2str(row - 1) '_' int2str(col)]);
            hold off;
            semilogy(Mat.H, 'LineWidth', LW);
            hold on;
            semilogy(Py.H, 'LineWidth', LW);
            print(f, ['raw_values_row=' int2str(row) '_hfun=' hfun{1} '_' Solvers{1} '_vs_' Solvers{2} '.png'], '-dpng', '-r400');
            close all;
        end
    end
end
