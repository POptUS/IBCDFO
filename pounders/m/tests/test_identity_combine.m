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

    X_0 = dfoxs(n, nprob, 1)';

    Low = -Inf(1, n);
    Upp = Inf(1, n);

    np_max = 2 * n + 1;
    nf_max = 1000;
    g_tol = 1e-13;
    delta_0 = 0.1;
    nfs = 1;

    hfun = @(F)F;
    combinemodels = @identity_combine;

    Ffun = @(x)calfun_wrapper_y(x, BenDFO, 'smooth');
    F_0 = Ffun(X_0);
    xk_in = 1;

    printf = 1;
    spsolver = 1;

    Prior.xk_in = xk_in;
    Prior.X_0 = X_0;
    Prior.F_init = F_0;
    Prior.nfs = nfs;

    Options.hfun = hfun;
    Options.combinemodels = combinemodels;
    Options.spsolver = spsolver;
    Options.printf = printf;

    Model.np_max = np_max;

    [X, F, hf, flag, xk_best] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, 1, Low, Upp, Prior, Options, Model);

    path(oldpath);

end

function [y] = calfun_wrapper_y(x, struct, probtype)
    [y, ~, ~] = calfun(x, struct, probtype);
end
