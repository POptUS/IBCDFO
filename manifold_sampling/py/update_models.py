import numpy as np
from ibcdfo.pounders import bmpts, formquad

from .call_user_scripts import call_user_scripts


def update_models(hfun, Ffun, n, p, nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, geometry_pt_flag, Hash, tol, L, U):
    Cres = F[xkin, :]
    Res = np.zeros(F.shape)  # Stores the residuals for model updates

    # for i in range(nf):
    #     D = X[i] - X[xkin]
    #     for j in range(len(Cres)):
    #         Res[i, j] = (F[i, j] - Cres[j]) - 0.5 * np.dot(D, np.dot(Hres[:, :, j], D.T))
    # Res2 = Res.copy()

    D = np.zeros((nf, X.shape[1]))
    D[:, :] = X[:nf, :] - X[xkin]
    for i, Di in enumerate(D):
        Res[i, :] = (F[i, :] - Cres[:]) - 0.5 * np.einsum("i,ijk,j->k", Di, Hres, Di)

    # assert(np.all(np.abs(Res2 - Res) < 1.0e-8))

    Mdir, mp, valid, Gres, Hresdel, _ = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xkin, fq_pars["npmax"], fq_pars["Par"], 0)

    # Evaluate geometry points
    if mp < n and geometry_pt_flag:  # Must obtain and evaluate bounded geometry points
        Mdir, mp = bmpts(X[xkin], Mdir[: n - mp], L, U, delta, fq_pars["Par"][2])

        for i in range(min(n - mp, nfmax - nf)):
            nf, X, F, h, Hash, _ = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + Mdir[i, :], tol, L, U, 1)
            Di = Mdir[i, :]

            # for j in range(p):
            #     Res[nf, j] = (F[nf, j] - Cres[j]) - 0.5 * np.dot(Di, np.dot(Hres[:, :, j], Di.T))
            # Res2 = Res.copy()

            Res[nf, :] = (F[nf, :] - Cres[:]) - 0.5 * np.einsum("i,ijk,j->k", Di, Hres, Di)

            # assert np.all(np.abs(Res[nf] - Res2[nf]) < 1.0e-8)

        _, _, valid, Gres, Hresdel, _ = formquad(X[: nf + 1], Res[: nf + 1], delta, xkin, fq_pars["npmax"], fq_pars["Par"], 0)

        if len(Gres) == 0:
            return valid, Gres, Hres, X, F, h, nf, Hash

    if len(Gres):  # We'll be doing evaluations; Hres will be updated after that
        Hres = Hres + Hresdel

    return valid, Gres, Hres, X, F, h, nf, Hash
