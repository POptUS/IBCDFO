function test_identity_combine()
    % This is a matlab test trying to reproduce the behavior of the python test
    % pounders/py/tests/test_showing_odd_pounders_formquad_interpolation_behavior.py
    % but doesn't fully do so due to differences in matlab/python.

    [here_path, ~, ~] = fileparts(mfilename('fullpath'));
    oldpath = addpath(fullfile(here_path, '..'));
    addpath(fullfile(here_path, '..', 'general_h_funs'));

    load dfo.dat;

    nprob = dfo(11, 1);
    n = dfo(11, 2);
    m = dfo(11, 3);

    BenDFO.nprob = nprob;
    BenDFO.m = m;
    BenDFO.n = n;

    X0 = dfoxs(n, nprob, 1)';

    Low = -Inf(1, n);
    Upp = Inf(1, n);

    npmax = 2 * n + 1;
    nf_max = 1000;
    gtol = 1e-13;
    delta = 0.1;
    nfs = 1;

    hfun = @(F)F;
    combinemodels = @identity_combine;

    Ffun = @(x)calfun_wrapper_y(x, BenDFO, 'smooth');
    F0 = Ffun(X0);
    xkin = 1;

    printf = 1;
    spsolver = 1;

    [X, F, flag, xk_best] = pounders(Ffun, X0, n, npmax, nf_max, gtol, delta, nfs, 1, F0, xkin, Low, Upp, printf, spsolver, hfun, combinemodels);

    path(oldpath);

end

function [y] = calfun_wrapper_y(x, struct, probtype)
    [y, ~, ~] = calfun(x, struct, probtype);
end
