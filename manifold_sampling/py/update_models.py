import numpy as np


def update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, geometry_pt_flag, Hash, tol, L, U):
    Cres = F[xkin, :]
    Res = np.zeros((F.shape, F.shape))

    for i in np.arange(1, nf + 1).reshape(-1):
        D = X[i, :] - X[xkin, :]
        for j in np.arange(1, len(Cres) + 1).reshape(-1):
            Res[i, j] = (F[i, j] - Cres[j]) - 0.5 * D * Hres[:, :, j] * np.transpose(D)

    Mdir, np, valid, Gres, Hresdel, __ = formquad(X[np.arange(1, nf + 1), :], Res[np.arange(1, nf + 1), :], delta, xkin, fq_pars.npmax, fq_pars.Par, 0)
    # Evaluate geometry points
    if np < n and geometry_pt_flag:
        Mdir, np = bmpts(X[xkin, :], Mdir[np.arange(1, n - np + 1), :], L, U, delta, fq_pars.Par[3])
        for i in np.arange(1, np.amin(n - np, nfmax - nf) + 1).reshape(-1):
            nf, X, F, h, Hash = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + Mdir[i, :], tol, L, U, 1)
            D = Mdir[i, :]
            for j in np.arange(1, p + 1).reshape(-1):
                Res[nf, j] = (F[nf, j] - Cres[j]) - 0.5 * D * Hres[:, :, j] * np.transpose(D)
        __, __, valid, Gres, Hresdel, __ = formquad(X[np.arange(1, nf + 1), :], Res[np.arange(1, nf + 1), :], delta, xkin, fq_pars.npmax, fq_pars.Par, 0)
        if len(Gres) == 0:
            return valid, Gres, Hres, X, F, h, nf, Hash

    if not len(Gres) == 0:
        Hres = Hres + Hresdel

    return valid, Gres, Hres, X, F, h, nf, Hash
