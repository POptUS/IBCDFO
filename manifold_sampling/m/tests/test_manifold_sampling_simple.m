% This tests a single-manifold case for manifold sampling. 

function [] = test_manifold_sampling_simple()

[here_path, ~, ~] = fileparts(mfilename('fullpath'));
oldpath = addpath(fullfile(here_path, '..'));
addpath(fullfile(here_path, '..', 'h_examples'));

nfmax = 300;
factor = 10;

hfun = @sum_squared;
subprob_switch = 'linprog';

load dfo.dat;

for row = [1, 2]
    nprob = dfo(row, 1);
    n = dfo(row, 2);
    m = dfo(row, 3);
    factor_power = dfo(row, 4);

    BenDFO.nprob = nprob;
    BenDFO.n = n;
    BenDFO.m = m;

    LB = -5000 * ones(1, n);
    UB = 5000 * ones(1, n);

    xs = dfoxs(n, nprob, factor^factor_power);

    Ffun = @(x)calfun_wrapper(x, BenDFO, 'smooth');
    x0 = xs';

    [~, ~, hF, xkin, ~] = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch);
    if row == 1 || row == 2
        assert(hF(xkin) <= 36.0 + 1e-8, "Not within 1e-8 of known minima")
    end
end
path(oldpath);
end

function [fvec] = calfun_wrapper(x, struct, probtype)
    [~, fvec, ~] = calfun(x, struct, probtype);
    fvec = fvec';
end