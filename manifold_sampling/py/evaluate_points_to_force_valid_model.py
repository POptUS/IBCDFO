import numpy as np


def evaluate_points_to_force_valid_model(n, nf, xkin, delta, X, F, h, gentype, Mdir, np, hfun, Ffun, Hash, fq_pars, tol, nfmax, L, U):
    # global nprob Qs zs bs p x0 nfmax h_activity_tol row_in_dfo_dat s inst
    # A.n=n; A.nf=nf; A.xkin=xkin; A.delta=delta; A.X=X; A.F=F; A.h=h; A.gentype=gentype; A.Mdir=Mdir; A.np=np; A.hfun=hfun; A.Ffun=Ffun; A.Hash=Hash; A.fq_pars=fq_pars; A.tol=tol; A.nfmax=nfmax; A.L=L; A.U=U;

    # Evaluate model-improving points to pick best one
    # ! May eventually want to normalize Mdir first for infty norm
    # Plus directions
    # *** Dec 2016: THIS ASSUMES UNCONSTRAINED, proceed with caution
    Mdir1, np1 = bmpts(X[xkin, :], Mdir[np.arange(1, n - np + 1), :], L, U, delta, fq_pars.Par[3])
    # Res = zeros(n-np, 1);
    for i in np.arange(1, n - np1 + 1).reshape(-1):
        # if ~all(isinf(L)) || ~all(isinf(U))
        #     D = Mdir1(i, :);
        #     Res(i, 1) = D*(g_k+.5*H_k*D');
        #     if Res(i, 1)> D*(-g_k+.5*H_k*D');
        #         Mdir1(i, :)=-Mdir1(i, :); #neg dir predicted 2b better
        #     end
        # end
        Xsp = Mdir1[i, :]
        # Only do this evaluation if the point is new and nf < nfmax
        if not ismember(X[xkin, :] + Xsp, X[np.arange(1, nf + 1), :], "rows") and nf < nfmax:
            nf, X, F, h, Hash = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + Xsp, tol, L, U, 1)

        __, __, valid = formquad(X[np.arange(1, nf + 1), :], F[np.arange(1, nf + 1), :], delta, xkin, fq_pars.npmax, fq_pars.Par, 1)
        if not valid and nf < nfmax:
            print(nf)
            print(gentype)
            print("Proceeding with nonvalid model! Report this to Stefan in Alg1")
            # uuid = char(java.util.UUID.randomUUID);
            # global mw_prob_num hfun
            # save(['first_failure_for_row_in_dfo_dat=' int2str(mw_prob_num) '_hfun=' func2str(hfun{1}) ], 'A');
            # error('a')

    return X, F, h, nf, Hash
