import numpy as np
from call_user_scripts import call_user_scripts
from ibcdfo.pounders import bmpts, formquad


def evaluate_points_to_force_valid_model(n, nf, xkin, delta, X, F, h, gentype, Mdir, mp, hfun, Ffun, Hash, fq_pars, tol, nfmax, L, U, Gfun=None, G=None):
    # Evaluate model-improving points to pick best one
    # ! May eventually want to normalize Mdir first for infty norm
    # Plus directions
    # *** Dec 2016: THIS ASSUMES UNCONSTRAINED, proceed with caution
    Mdir1, np1 = bmpts(X[xkin, :], Mdir[: n - mp + 1], L, U, delta, fq_pars["Par"][3])
    # Res = zeros(n-np, 1);
    for i in range(n - np1):
        # if ~all(isinf(L)) || ~all(isinf(U))
        #     D = Mdir1(i, :);
        #     Res(i, 1) = D*(g_k+.5*H_k*D');
        #     if Res(i, 1)> D*(-g_k+.5*H_k*D');
        #         Mdir1(i, :)=-Mdir1(i, :); #neg dir predicted 2b better
        #     end
        # end
        Xsp = Mdir1[i, :]
        # Only do this evaluation if the point is new and nf < nfmax
        if not np.any(np.all(X[xkin, :] + Xsp == X[: nf + 1], axis=1)) and nf + 1 < nfmax:
            if Gfun is None:
                nf, X, F, h, Hash, _ = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + Xsp, tol, L, U, 1)
            else:
                nf, X, F, G, h, Hash, _ = call_user_scripts2(nf, X, F, G, h, Hash, Ffun, Gfun, hfun, X[xkin, :] + Xsp, tol, L, U, 1)

    valid = formquad(X[: nf + 1], F[: nf + 1], delta, xkin, fq_pars["npmax"], fq_pars["Par"], 1)[2]
    if not valid and nf + 1 < nfmax:
        print(nf)
        print(gentype)
        print("Proceeding with nonvalid model! Report this to Stefan in Alg1")
        # uuid = char(java.util.UUID.randomUUID);
        # global mw_prob_num hfun
        # save(['first_failure_for_row_in_dfo_dat=' int2str(mw_prob_num) '_hfun=' func2str(hfun{1}) ], 'A');
        # error('a')

    if Gfun is None:
        return X, F, h, nf, Hash
    else:
        return X, F, G, h, nf, Hash
