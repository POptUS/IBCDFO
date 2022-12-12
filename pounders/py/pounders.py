import numpy as np
from checkinputss import checkinputss
from formquad import formquad
from bmpts import bmpts
from bqmin import bqmin

import sys


def pounders(fun, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver, hfun=None, combinemodels=None):
    # POUNDERS: Practical Optimization Using No Derivatives for sums of Squares
    #   [X,F,flag,xkin] = ...
    #        pounders(fun,X0,n,mpmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,printf)
    #
    # This code minimizes a blackbox function, solving
    # min { f(X)=sum_(i=1:m) F_i(x)^2, such that L_j <= X_j <= U_j, j=1,...,n }
    # where the user-provided F is specified in the handle fun. Evaluation of
    # this F must result in the return of a 1-by-m row vector. Bounds must be
    # specified in U and L but can be set to L=-Inf(1,n) and U=Inf(1,n) if the
    # unconstrained solution is desired. The algorithm will not evaluate F
    # outside of these bounds, but it is possible to take advantage of function
    # values at infeasible X if these are passed initially through (X0,F0).
    # In each iteration, the algorithm forms an interpolating quadratic model
    # of the function and minimizes it in an infinity-norm trust region.
    #
    # This software comes with no warranty, is not bug-free, and is not for
    # industrial use or public distribution.
    # Direct requests and bugs to wild@mcs.anl.gov.
    # A technical report/manual is forthcoming, a brief description is in
    # Nuclear Energy Density Optimization. Phys. Rev. C, 82:024313, 2010.
    #
    # --INPUTS-----------------------------------------------------------------
    # fun     [f h] Function handle so that fun(x) evaluates F (@calfun)
    # X0      [dbl] [max(nfs,1)-by-n] Set of initial points  (zeros(1,n))
    # n       [int] Dimension (number of continuous variables)
    # mpmax   [int] Maximum number of interpolation points (>n+1) (2*n+1)
    # nfmax   [int] Maximum number of function evaluations (>n+1) (100)
    # gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
    # delta   [dbl] Positive trust region radius (.1)
    # nfs     [int] Number of function values (at X0) known in advance (0)
    # m       [int] Number of residual components
    # F0      [dbl] [nfs-by-m] Set of known function values  ([])
    # xkin    [int] Index of point in X0 at which to start from (1)
    # L       [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
    # U       [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
    # printf  [log] 0 No printing to screen
    #               1 Debugging level of output to screen (default)
    #               2 More verbose screen output
    # spsolver [int] Trust-region subproblem solver flag
    #
    # Optionally, a user can specify and outer-function that maps the the elements
    # of F to a scalar value (to be minimized). Doing this also requires a function
    # handle (combinemodels) that tells pounders how to map the linear and
    # quadratic terms from the residual models into a single quadratic TRSP model.
    #
    # hfun           [f h] Function handle for mapping output from F
    # combinemodels  [f h] Function handle for combine residual models
    #
    # --OUTPUTS----------------------------------------------------------------
    # X       [dbl] [nfmax+nfs-by-n] Locations of evaluated points
    # F       [dbl] [nfmax+nfs-by-m] Function values of evaluated points
    # flag    [dbl] Termination criteria flag:
    #               = 0 normal termination because of grad,
    #               > 0 exceeded nfmax evals,   flag = norm of grad at final X
    #               = -1 if input was fatally incorrect (error message shown)
    #               = -2 if a valid model produced X[nf] == X[xkin] or (mdec == 0, Fs[nf] == Fs[xkin])
    # xkin    [int] Index of point in X representing approximate minimizer

    if hfun is None:
        hfun = lambda F: np.sum(F**2)
        sys.path.append('../../')
        from general_h_funs import leastsquares as combinemodels

    # choose your spsolver
    if spsolver == 2:
        sys.path.append("../../minq/minq5/python/")
        from minqsw import minqsw
    elif spsolver == 3:
        sys.path.append("../../minq/minq8/python/")
        from minq8 import minq8

    [flag, X0, mpmax, F0, L, U] = checkinputss(fun, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U)
    if flag == -1:
        X = []
        F = []
        return [X, F, flag, xkin]
    maxdelta = min(0.5 * np.min(U - L), (10**3) * delta)
    mindelta = min(delta * (10**-13), gtol / 10)
    gam0 = 0.5
    gam1 = 2
    eta1 = 0.05
    Par = np.zeros(4)
    Par[0] = np.sqrt(n)
    Par[1] = max(10, np.sqrt(n))
    Par[2] = 10**-3
    Par[3] = 0.001
    eps = np.finfo(float).eps  # Define machine epsilon
    if printf:
        print('  nf   delta    fl  np       f0           g0       ierror')
        progstr = '%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n'  # Line-by-line
    if nfs == 0:
        X = np.vstack((X0, np.zeros((nfmax - 1, n))))
        F = np.zeros((nfmax, m))
        nf = 0  # in Matlab this is 1
        F[nf] = fun(X[nf])
        if np.any(np.isnan(F[nf])):
            print("A NaN was encountered in an objective evaluation. Exiting.")
            X = X[:nf]
            F = F[:nf]
            flag = -3;
            return
        if printf:
            print('%4i    Initial point  %11.5e\n' % (nf, np.sum(F[nf, :] ** 2)))
    else:
        X = np.vstack((X0[0 : max(1, nfs), :], np.zeros((nfmax, n))))
        F = np.vstack((F0[0:nfs, :], np.zeros((nfmax, m))))
        nf = nfs - 1
        nfmax = nfmax + nfs
    Fs = np.zeros(nfmax + nfs)
    for i in range(nf + 1):
        Fs[i] = hfun(F[i])
    Res = np.zeros(np.shape(F))
    Cres = F[xkin]
    Hres = np.zeros((n, n, m))
    while nf + 1 < nfmax:
        #  1a. Compute the interpolation set.
        for i in range(nf + 1):
            D = X[i] - X[xkin]
            Res[i, :] = (F[i, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
        [Mdir, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xkin, mpmax, Par, 0)
        if mp < n:
            [Mdir, mp] = bmpts(X[xkin], Mdir[0 : n - mp, :], L, U, delta, Par[2])
            for i in range(int(min(n - mp, nfmax - (nf + 1)))):
                nf += 1
                X[nf] = np.minimum(U, np.maximum(L, X[xkin] + Mdir[i, :]))
                F[nf] = fun(X[nf])
                if np.any(np.isnan(F[nf])):
                    print("A NaN was encountered in an objective evaluation. Exiting.")
                    X = X[:nf]
                    F = F[:nf]
                    flag = -3;
                    return
                Fs[nf] = hfun(F[nf])
                if printf:
                    print('%4i   Geometry point  %11.5e\n' % (nf, Fs[nf]))
                D = Mdir[i, :]
                Res[nf, :] = (F[nf, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
            if nf + 1 >= nfmax:
                break
            [_, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xkin, mpmax, Par, False)
            # [~,np,valid,Gres,Hresdel,Mind] = ...
            # formquad(X(1:nf,:),Res(1:nf,:),delta,xkin,mpmax,Par,0);
        #  1b. Update the quadratic model
        Cres = F[xkin]
        Hres = Hres + Hresdel
        c = Fs[xkin]
        G, H = combinemodels(Cres, Gres, Hres)
        ng = np.linalg.norm(G * (np.int64(X[xkin] > L) * np.int64(G.T > 0) + np.int64(X[xkin] < U) * np.int64(G.T < 0)).T, 2)
        if printf >= 2:
            IERR = np.zeros(len(Mind))
            for i in range(len(Mind)):
                D = X[Mind[i]] - X[xkin]
                IERR[i] = (c - Fs[Mind[i]]) + [D @ (G + 0.5 * H @ D)]
            jerr = np.zeros((len(Mind), m))
            for i in range(len(Mind)):
                D = X[Mind[i]] - X[xkin]
                for j in range(m):
                    jerr[i, j] = (Cres[j] - F[Mind[i], j]) + D @ (Gres[:, j] + 0.5 * Hres[:, :, j] @ D)
            print(jerr)
            # input("Enter a key and press Enter to continue\n") - Don't uncomment when using Pytest with test_pounders.py
            ierror = np.linalg.norm(IERR / np.abs(Fs[Mind]), np.inf)
            print(progstr % (nf, delta, valid, mp, Fs[xkin], ng, ierror))
        # 2. Critically test invoked if the projected model gradient is small
        if ng < gtol:
            delta = max(gtol, np.max(np.abs(X[xkin])) * eps)
            [Mdir, _, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xkin, mpmax, Par, 1)
            if not valid:
                [Mdir, mp] = bmpts(X[xkin], Mdir, L, U, delta, Par[2])
                for i in range(min(n - mp, nfmax - (nf + 1))):
                    nf += 1
                    X[nf] = np.minimum(U, np.maximum(L, X[xkin] + Mdir[i, :]))
                    F[nf] = fun(X[nf])
                    if np.any(np.isnan(F[nf])):
                        print("A NaN was encountered in an objective evaluation. Exiting.")
                        X = X[:nf]
                        F = F[:nf]
                        flag = -3;
                        return
                    Fs[nf] = hfun(F[nf])
                    if printf:
                        print('%4i   Critical point  %11.5e\n' % (nf, Fs[nf]))
                if nf + 1 >= nfmax:
                    break
                # Recalculate gradient based on a MFN model
                [_, _, valid, Gres, Hres, Mind] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xkin, mpmax, Par, 0)
                G, H = combinemodels(Cres, Gres, Hres)
                ng = np.linalg.norm(G * (np.int64(X[xkin] > L) * np.int64(G.T > 0) + np.int64(X[xkin] < U) * np.int64(G.T < 0)).T, 2)
            if ng < gtol:
                if printf:
                    print('g is sufficiently small')
                X = X[: nf + 1, :]
                F = F[: nf + 1, :]
                flag = 0
                return [X, F, flag, xkin]

        # 3. Solve the subproblem min{G.T * s + 0.5 * s.T * H * s : Lows <= s <= Upps }
        Lows = np.maximum(L - X[xkin], -delta * np.ones((np.shape(L))))
        Upps = np.minimum(U - X[xkin], delta * np.ones((np.shape(U))))
        if spsolver == 1:  # Stefan's crappy 10line solver
            [Xsp, mdec] = bqmin(H, G, Lows, Upps)
        elif spsolver == 2:  # Arnold Neumaier's minq5
            [Xsp, mdec, _, _] = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        elif spsolver == 3:  # Arnold Neumaier's minq8
            [Xsp, mdec, _, _] = minq8(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        Xsp = Xsp.squeeze()
        step_norm = np.linalg.norm(Xsp, np.inf)

        # 4. Evaluate the function at the new point
        if (step_norm >= 0.01 * delta or valid) and not (mdec == 0 and not valid):
            Xsp = np.minimum(U, np.maximum(L, X[xkin] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (U[i] - Xsp[i] < eps * abs(U[i])) and (U[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = U[i]
                    print('eps project!')
                elif (Xsp[i] - L[i] < eps * abs(L[i])) and (L[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = L[i]
                    print('eps project!')

            if mdec == 0 and valid and np.array_equiv(Xsp, X[xkin]):
                print('Terminating because mdec == 0 with a valid model and no change in Xsp')
                X = X[: nf + 1, :]
                F = F[: nf + 1, :]
                flag = -2
                return [X, F, flag, xkin]

            nf += 1
            X[nf] = Xsp
            F[nf] = fun(X[nf])
            if np.any(np.isnan(F[nf])):
                print("A NaN was encountered in an objective evaluation. Exiting.")
                X = X[:nf]
                F = F[:nf]
                flag = -3;
                return
            Fs[nf] = hfun(F[nf])

            if mdec != 0:
                rho = (Fs[nf] - Fs[xkin]) / mdec
            else:
                if Fs[nf] == Fs[xkin]:
                    print('Terminating because mdec == 0 with a valid model and Fs(nf) == Fs(xkin)')
                    X = X[: nf + 1, :]
                    F = F[: nf + 1, :]
                    flag = -2
                    return [X, F, flag, xkin]
                else:
                    rho = np.inf * np.sign(Fs[nf] - Fs[xkin])

            # 4a. Update the center
            if (rho >= eta1) or (rho > 0 and valid):
                # Update model to reflect new center
                Cres = F[xkin]
                xkin = nf  # Change current center
            # 4b. Update the trust-region radius:
            if (rho >= eta1) and (step_norm > 0.75 * delta):
                delta = min(delta * gam1, maxdelta)
            elif valid:
                delta = max(delta * gam0, mindelta)
        else:  # Don't evaluate f at Xsp
            rho = -1  # Force yourself to do a model-improving point
            if printf:
                print('Warning: skipping sp soln!-----------')
        # 5. Evaluate a model-improving point if necessary
        if not valid and (nf + 1 < nfmax) and (rho < eta1):  # Implies xkin, delta unchanged
            # Need to check because model may be valid after Xsp evaluation
            [Mdir, mp, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xkin, mpmax, Par, 1)
            if not valid:  # ! One strategy for choosing model-improving point:
                # Update model (exists because delta & xkin unchanged)
                for i in range(nf + 1):
                    D = X[i, :] - X[xkin]
                    Res[i, :] = (F[i, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
                [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xkin, mpmax, Par, False)
                Hres = Hres + Hresdel
                # Update for modelimp; Cres unchanged b/c xkin unchanged
                G, H = combinemodels(Cres, Gres, Hres)
                # Evaluate model-improving points to pick best one
                # May eventually want to normalize Mdir first for infty norm
                # Plus directions
                [Mdir1, mp1] = bmpts(X[xkin], Mdir[0 : n - mp, :], L, U, delta, Par[2])
                for i in range(n - mp1):
                    D = Mdir1[i, :]
                    Res[i, 0] = D @ (G + 0.5 * H @ D.T)
                b = np.argmin(Res[: n - mp1, 0:1])
                a1 = np.min(Res[: n - mp1, 0:1])
                Xsp = Mdir1[b, :]
                # Minus directions
                [Mdir1, mp2] = bmpts(X[xkin], -Mdir[0 : n - mp, :], L, U, delta, Par[2])
                for i in range(n - mp2):
                    D = Mdir1[i, :]
                    Res[i, 0] = D @ (G + 0.5 * H @ D.T)
                b = np.argmin(Res[: n - mp2, 0:1])
                a2 = np.min(Res[: n - mp2, 0:1])
                if a2 < a1:
                    Xsp = Mdir1[b, :]
                nf += 1
                X[nf] = np.minimum(U, np.maximum(L, X[xkin] + Xsp))  # Temp safeguard
                F[nf] = fun(X[nf])
                if np.any(np.isnan(F[nf])):
                    print("A NaN was encountered in an objective evaluation. Exiting.")
                    X = X[:nf]
                    F = F[:nf]
                    flag = -3;
                    return
                Fs[nf] = hfun(F[nf])
                if printf:
                    print('%4i   Model point     %11.5e\n' % (nf, Fs[nf]))
                if Fs[nf] < Fs[xkin]:  # ! Eventually check stuff decrease here
                    if printf:
                        print('**improvement from model point****')
                    # Update model to reflect new base point
                    D = X[nf] - X[xkin]
                    xkin = nf  # Change current center
                    Cres = F[xkin]
                    # Don't actually use
                    for j in range(m):
                        Gres[:, j] = Gres[:, j] + Hres[:, :, j] @ D.T
    if printf:
        print('Number of function evals exceeded')
    flag = ng
    return [X, F, flag, xkin]
