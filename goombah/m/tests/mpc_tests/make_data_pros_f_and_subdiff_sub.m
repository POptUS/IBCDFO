addpath('export_fig'); % https://github.com/altmany/export_fig.git

bendfo_root = '../../../../../BenDFO/';
dfo = load([bendfo_root 'data/dfo.dat']);
nfmax_c = 20; % 100;

num_seeds = 1;
solver_names = {'MS-D', 'GOOMBAH', 'MS-P', 'GOOMBAH+MS-P'}; % Used when saving filenames for ease of reference
num_solvers = length(solver_names);
ind_in_H = [1 2 3 4];

hfuns = {@pw_minimum_squared, @pw_maximum_squared, @censored_L1_loss, @piecewise_quadratic};
constrs = [false]; % [true false];
% hfuns = {@pw_minimum_squared};
% hfuns = {@pw_maximum_squared};
% hfuns = {@censored_L1_loss};
% hfuns = {@piecewise_quadratic};

for hfun = hfuns
    for constr = constrs
        Hist_h = inf(nfmax_c * (max(dfo(:, 2) + 1)), 53 * num_seeds, num_solvers);
        Hist_norm = inf(nfmax_c * (max(dfo(:, 2) + 1)), 53 * num_seeds, num_solvers);

        original_size = size(Hist_h);

        prob = 1;
        prob_dim = [];

        for mw_prob_num = 1:53
            n = dfo(mw_prob_num, 2);
            nfmax = nfmax_c * (n + 1);

            for seed = 1:num_seeds
                for s = 1:num_solvers
                    s1 = ind_in_H(s);
                    processed_filename = ['processed_results/processed_' solver_names{s} '_prob=' int2str(mw_prob_num) '_seed=' ...
                                          int2str(seed) '_' func2str(hfun{1}) '_nfmax_c=' num2str(nfmax_c) '_constr=' int2str(constr) '_alt.mat'];
                    if ~exist(processed_filename, 'file') || dir(processed_filename).bytes == 0
                        processed_filename;
                        continue
                    end

                    clear A;
                    A = load(processed_filename);

                    for i = 1:min(nfmax, size(A.Hist_norm, 1))
                        Hist_h(i, prob, s) = A.Hist_h(i, s1);
                        Hist_norm(i, prob, s) = A.Hist_norm(i, s1);
                    end
                end
                prob_dim(prob) = n;
                prob = prob + 1;
            end
        end
        assert(all(original_size == size(Hist_h)), "Hist_h grew and probably didn't have infs in it like it should");

        Solvers = solver_names;
        Solvers{1} = '\texttt{MS-D}';
        Solvers{2} = '\texttt{GOOMBAH} w/o \texttt{MS-P}';
        Solvers{3} = '\texttt{MS-P}';
        Solvers{4} = '\texttt{GOOMBAH}';

        LW = 2;
        FS = 14;
        Label_FS = 14;
        % all_done = ~sum(isinf(squeeze(Hist_h(1, :, :))), 2);
        all_done = 1:size(Hist_h, 2);
        to_plot = 1:num_solvers;
        % cases = {'norm', 'h'};
        cases = {'norm'};
        if constr
            neworder = [4 2 3];
        else
            neworder = [4 2 3 1];
        end
        for tau = logspace(-5, -1, 3)
            for c = cases
                ci = c{:};
                close all;
                f = figure;
                switch ci
                    case 'norm'
                        [h, rawT] = data_profile_subdiff(Hist_norm(:, all_done, to_plot), prob_dim(all_done) + 1, tau, 0);
                    case 'h'
                        [h, rawT] = data_profile(Hist_h(:, all_done, to_plot), prob_dim(all_done) + 1, tau, 0);
                end

                xlim([0, nfmax_c]);

                set(findall(f, 'type', 'line'), 'LineWidth', 2);
                set(gca, 'FontSize', FS);

                % title(['tau=' num2str(tau) ' constr=' int2str(constr) ' prob=' strrep(func2str(hfun{1}), '_', '-')], 'FontSize', 12)
                % subtitle(' ','FontSize',5)

                [~, hl] = legend(h([neworder]), Solvers([neworder]), 'Location', 'SouthEast', 'interpreter', 'latex');
                objhl = findobj(hl, 'type', 'line'); % // objects of legend of type line
                set(objhl, 'Markersize', 10); % // set marker size as desired

                xlabel('Function evaluations divided by $(n_p+1)$', 'FontSize', Label_FS, 'Interpreter', 'LaTeX');
                ylabel('Fraction of problems', 'FontSize', Label_FS, 'Interpreter', 'LaTeX');

                export_fig(f, ['data_prof_' ci '_tau=' num2str(tau) '_nfmax=' int2str(nfmax_c) 'nplus1_ns=' int2str(length(to_plot)) '_' func2str(hfun{:}) '_constr=' int2str(constr) '_alt.png'], '-transparent', '-r300');
            end
        end
    end
end
