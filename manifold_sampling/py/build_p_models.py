import numpy as np


def build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U):
    n = X.shape[2 - 1]
    p = F.shape[2 - 1]
    valid, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U)
    # Evaluate model-improving points if necessary
    if not valid and not len(Gres) == 0:
        Mdir, np, valid = formquad(X[np.arange(1, nf + 1), :], F[np.arange(1, nf + 1), :], delta, xkin, fq_pars.npmax, fq_pars.Par, 1)
        if valid:
            raise Exception("what to do here")
        X, F, h, nf, Hash = evaluate_points_to_force_valid_model(n, nf, xkin, delta, X, F, h, tol.gentype, Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nfmax, L, U)
        __, Gres, Hres, X, F, h, nf, Hash = update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, 1, Hash, tol, L, U)

    return Gres, Hres, X, F, h, nf, Hash

    return Gres, Hres, X, F, h, nf, Hash
