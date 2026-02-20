% Compare two POUNDERS4py runs: serial vs concurrent
% Loads from:
%   ./benchmark_results/serial/
%   ./benchmark_results/concurrent/
%
% Robust to .mat variable naming differences: it auto-detects the right struct
% inside each MAT file (expects a struct containing field 'H').

clear all;

addpath('../../../../BenDFO/profiling/');
addpath('../../../../BenDFO/data/');
load dfo.dat;

% --------------------- user params ---------------------
method   = 'pounders';
gtol     = 1e-13; %#ok<NASGU>
spsolver = 2;

LW       = 2;
FS       = 15;
Label_FS = 15;

np       = 53;
prob_dim = dfo(:, 2);

% Plot processing mode per solver:
%   1  -> raw H
%  -1  -> take min over blocks of 5
%  else-> use Group processing
plot_pars = [1, 1];

% --------------------- directories ---------------------
base_dir = './benchmark_results';
solver_dirs = { ...
    fullfile(base_dir, 'serial'), ...
    fullfile(base_dir, 'concurrent') ...
};
Solvers = {[method '-serial'], [method '-concurrent']};
ns = numel(solver_dirs);    % must be 2
assert(ns == 2);

% --------------------- load data -----------------------
countm = zeros(1, ns);
M = cell(1, ns);

for s = 1:ns
    for row = 1:np
        prob = row - 1;

        % Match:
        %   pounders4py_serial_nf_max=..._prob=..._spsolver=2.mat
        %   pounders4py_conc_nf_max=..._prob=..._spsolver=2.mat
        pat = sprintf('pounders4py_*_nf_max=*_prob=%d_spsolver=%d.mat', prob, spsolver);
        listing = dir(fullfile(solver_dirs{s}, pat));

        if isempty(listing)
            error('Missing file for prob=%d in %s (pattern: %s)', prob, solver_dirs{s}, pat);
        end
        if numel(listing) > 1
            warning('Multiple files matched for prob=%d in %s; using %s', prob, solver_dirs{s}, listing(1).name);
        end

        filename = fullfile(listing(1).folder, listing(1).name);
        S = load(filename);

        % Select the right struct from this MAT-file, robustly
        run = pick_run_struct(S, method, prob, filename);

        % Parse nf_max from the filename and store it
        tok = regexp(listing(1).name, 'nf_max=(\d+)_prob=', 'tokens', 'once');
        if ~isempty(tok)
            run.nf_max = str2double(tok{1});
        else
            run.nf_max = length(run.H);
        end

        countm(s) = countm(s) + 1;
        M{s}{countm(s)} = run;
    end

    assert(countm(s) == np, sprintf('Found only %d/%d files in %s', countm(s), np, solver_dirs{s}));
end

% --------------------- allocate H ----------------------
nfmax_all = zeros(np, ns);
for s = 1:ns
    for k = 1:np
        if isfield(M{s}{k}, 'nf_max') && ~isempty(M{s}{k}.nf_max)
            nfmax_all(k, s) = M{s}{k}.nf_max;
        else
            nfmax_all(k, s) = length(M{s}{k}.H);
        end
    end
end
nf_max_global = max(nfmax_all(:));

H = inf(nf_max_global + 100, np, ns);

% --------------------- fill H --------------------------
for k = 1:np
    for s = 1:ns
        len = min([nfmax_all(k, s), length(M{s}{k}.H)]);

        if plot_pars(s) == 1
            H(1:len, k, s) = M{s}{k}.H(1:len);

        elseif plot_pars(s) == -1
            H(1:len, k, s) = M{s}{k}.H(1:len);
            for i = 1:floor(len / 5)
                H(i, k, s) = min(H([(i - 1) * 5 + 1:i * 5], k, s));
            end

        else
            if ~isfield(M{s}{k}, 'Group')
                error('plot_pars(%d) requires Group field, but M{%d}{%d} has none.', s, s, k);
            end
            G = process_groups(M{s}{k}.Group);
            for i = 0:max(G)
                H(i + 1, k, s) = min(M{s}{k}.H(G == i));
            end
        end
    end
end

% --------------------- plot data profiles --------------
to_plot = [1, 2];

for tau = logspace(-3, -1, 3)
    f = figure;
    h = data_profile(H(:, :, to_plot), prob_dim + 1, tau); %#ok<NASGU>

    set(findall(f, 'type', 'line'), 'LineWidth', LW);
    set(gca, 'FontSize', FS);

    legend(h, Solvers(to_plot), 'Location', 'SouthEast');
    xlabel('Batched function evaluations divided by (n+1)', 'FontSize', Label_FS);
    ylabel('Fraction of problems solved', 'FontSize', Label_FS);

    % Tight layout (as in your original)
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];

    print(f, ['Fvalue_data_tau=' num2str(tau) '_nf_max=' int2str(nf_max_global) '_' Solvers{to_plot(1)} '_vs_' Solvers{to_plot(2)} '.png'], '-dpng', '-r400');
    close all;
end

% ========================================================================
% Helper: pick the "run struct" inside a MAT-file
% ========================================================================
function run = pick_run_struct(S, method, prob, filename)
    fn = fieldnames(S);

    % 1) Try the old convention(s) first (if present)
    exact_keys = { ...
        [method '4py_' int2str(prob)], ...
        [method '4py_prob_' int2str(prob)], ...
        ['pounders4py_' int2str(prob)] ...
    };
    for i = 1:numel(exact_keys)
        k = exact_keys{i};
        if isfield(S, k) && isstruct(S.(k)) && isfield(S.(k), 'H')
            run = S.(k);
            return;
        end
    end

    % 2) Otherwise: collect struct variables that look like a run
    cand = {};
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isstruct(v) && isfield(v, 'H')
            cand{end+1} = fn{i}; %#ok<AGROW>
        end
    end

    if isempty(cand)
        % Nothing looks like a run; show what IS in the file
        msg = sprintf('No struct with field "H" found in %s.\nVariables present:\n', filename);
        for i = 1:numel(fn)
            v = S.(fn{i});
            msg = msg + sprintf('  - %s : %s\n', fn{i}, class(v));
        end
        error('%s', msg);
    end

    % 3) If exactly one candidate, use it
    if numel(cand) == 1
        run = S.(cand{1});
        return;
    end

    % 4) Prefer candidates whose name mentions the prob index
    %    (e.g., "..._prob=0..." or ends with "_0", etc.)
    prob_str = int2str(prob);
    hits = cand(contains(cand, ['prob' prob_str]) | endsWith(cand, ['_' prob_str]) | contains(cand, prob_str));

    if numel(hits) == 1
        run = S.(hits{1});
        return;
    end

    % 5) Still ambiguous: error with options (so you can adjust selection rule)
    msg = sprintf('Ambiguous run struct in %s.\nFound %d structs with field "H":\n', filename, numel(cand));
    for i = 1:numel(cand)
        msg = msg + sprintf('  - %s\n', cand{i});
    end
    error('%s', msg);
end
