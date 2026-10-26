% This tests goombah and goombah_wo_msp on a few BenDFO problems and nonsmooth
% h functions, using the LHS-based TRSP solver instead of GAMS.

function [] = test_goombah_lhs()

clear global h_activity_tol;

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
repo_root = fullfile(here_path, '..', '..', '..');
oldpath = addpath(fullfile(repo_root, 'goombah', 'm'));
addpath(fullfile(repo_root, 'goombah', 'm', 'subproblems'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm'));
addpath(fullfile(repo_root, 'manifold_sampling', 'm', 'general_nonsmooth_h_funs'));
addpath(fullfile(repo_root, 'pounders', 'm'));

factor = 10;
hfuns = {@h_pw_maximum_squared, @h_pw_minimum_squared};
GAMS_options.trsp = 'lhs';

load dfo.dat;

for row = [1, 7]
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    x0 = dfoxs(n, nprob, factor^factor_power)';
    nf_max = 10 * (n + 1);
    L = -inf(1, n);
    U = inf(1, n);

    for i = 1:length(hfuns)
        hfun = hfuns{i};

        rand('seed', 1000 * row + i);
        [X, F, h, xkin] = goombah(hfun, Ffun, nf_max, x0, L, U, GAMS_options, 'linprog');
        check_outputs(X, F, h, xkin);

        rand('seed', 1000 * row + i);
        [X, F, h, xkin] = goombah_wo_msp(hfun, Ffun, nf_max, x0, L, U, GAMS_options);
        check_outputs(X, F, h, xkin);
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
