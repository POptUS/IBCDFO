import os
import sys

import numpy as np

from pathlib import Path

from poptus import (
    LOG_LEVEL_BASIC, LOG_LEVEL_DEBUG_BASIC
)

from .bmpts import bmpts
from .bqmin import bqmin
from .checkinputss import checkinputss
from .formquad import formquad
from .prepare_outputs_before_return import prepare_outputs_before_return


def pounders(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, logger, spsolver=2, hfun=None, combinemodels=None):
    """
    POUNDERS: Practical Optimization Using No Derivatives for sums of Squares
      [X,F,flag,xkin] = ...
           pounders(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,logger)

    This code minimizes output from a structured blackbox function, solving
    min { f(X)=sum_(i=1:m) F_i(x)^2, such that L_j <= X_j <= U_j, j=1,...,n }
    where the user-provided blackbox F is specified in the handle fun. Evaluation
    of this F must result in the return of a 1-by-m row vector. Bounds must be
    specified in U and L but can be set to L=-Inf(1,n) and U=Inf(1,n) if the
    unconstrained solution is desired. The algorithm will not evaluate F
    outside of these bounds, but it is possible to take advantage of function
    values at infeasible X if these are passed initially through (X0,F0).
    In each iteration, the algorithm forms an interpolating quadratic model
    of the function and minimizes it in an infinity-norm trust region.

    This software comes with no warranty, is not bug-free, and is not for
    industrial use or public distribution.
    Direct requests and bugs to wild@mcs.anl.gov.
    A technical report/manual is forthcoming, a brief description is in
    Nuclear Energy Density Optimization. Phys. Rev. C, 82:024313, 2010.

    --INPUTS-----------------------------------------------------------------
    fun     [f h] Function handle so that fun(x) evaluates F (@calfun)
    X0      [dbl] [max(nfs,1)-by-n] Set of initial points  (zeros(1,n))
    n       [int] Dimension (number of continuous variables)
    npmax   [int] Maximum number of interpolation points (>=n+1) (2*n+1)
    nfmax   [int] Maximum number of function evaluations (>n+1) (100)
    gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
    delta   [dbl] Positive trust region radius (.1)
    nfs     [int] Number of function values (at X0) known in advance (0)
    m       [int] Number of residual components
    F0      [dbl] [nfs-by-m] Set of known function values  ([])
    xkin    [int] Index of point in X0 at which to start from (1)
    L       [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
    U       [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
    logger        Logger object derived from poptus.AbcLogger
    spsolver [int] Trust-region subproblem solver flag (2)

    Optionally, a user can specify and outer-function that maps the the elements
    of F to a scalar value (to be minimized). Doing this also requires a function
    handle (combinemodels) that tells pounders how to map the linear and
    quadratic terms from the residual models into a single quadratic TRSP model.

    hfun           [f h] Function handle for mapping output from F
    combinemodels  [f h] Function handle for combine residual models

    --OUTPUTS----------------------------------------------------------------
    X       [dbl] [nfmax+nfs-by-n] Locations of evaluated points
    F       [dbl] [nfmax+nfs-by-m] Function values of evaluated points
    flag    [dbl] Termination criteria flag:
                  = 0 normal termination because of grad,
                  > 0 exceeded nfmax evals,   flag = norm of grad at final X
                  = -1 if input was fatally incorrect (error message shown)
                  = -2 if a valid model produced X[nf] == X[xkin] or (mdec == 0, Fs[nf] == Fs[xkin])
                  = -3 error if a NaN was encountered
                  = -4 error in TRSP Solver
                  = -5 unable to get model improvement with current parameters
    xkin    [int] Index of point in X representing approximate minimizer
    """
    LOG_TAG = "POUNDerS"

    def log(msg):
        logger.log(LOG_TAG, msg, LOG_LEVEL_BASIC)

    def log_debug(msg, level):
        logger.log(LOG_TAG, msg, LOG_LEVEL_DEBUG_BASIC + level)

    def log_warning(msg):
        logger.warn(LOG_TAG, msg)

    def log_error(flag):
        # Based on discussions with Jeff, failure should be communicated by
        # this function returning an appropriate flag value rather than by
        # raising an exception.  Therefore this just prints.
        if flag == -4:
            msg = "A minq input error occurred. Exiting."
        elif flag == -3:
            msg = "A NaN was encountered in an objective evaluation. Exiting."
        elif flag == -2:
            msg = "Terminating because mdec == 0 with a valid model and no improvement from TRSP solution. Exiting."
        elif flag == -5:
            msg = "Unable to improve model with current Pars; try dividing Par[2:3] by 10. Exiting."
        elif flag == -1:
            msg = "Number of residuals in output of fun does not match supplied m. Exiting."
        elif flag > 0:
            msg = "Number of function evals exceeded. Exiting."
        logger.error(LOG_TAG, msg)

    if hfun is None:

        def hfun(F):
            return np.sum(F**2)

        from .general_h_funs import leastsquares as combinemodels

    # choose your spsolver
    if spsolver == 2:
        env_var = "POPTUS_MINQ_PATH"
        if not env_var in os.environ:
            msg = f"Set env var {env_var} to root of MINQ clone"
            logger.error(LOG_TAG, msg)
            raise RuntimeError(msg)
        minq_path = Path(os.environ[env_var]).resolve()
        if not minq_path.is_dir():
            msg = f"Set env var {env_var} to root of MINQ clone"
            logger.error(LOG_TAG, msg)
            raise RuntimeError(msg)
        minq_path = minq_path.joinpath("py", "minq5").resolve()
        assert minq_path.is_dir()
        sys.path.append(str(minq_path))
        from minqsw import minqsw

    [flag, X0, npmax, F0, L, U, xkin] = checkinputss(fun, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U)
    if flag == -1:
        X = []
        F = []
        return X, F, flag, xkin
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
    log("  nf   delta    fl  np       f0           g0       ierror")
    progstr = "%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e"  # Line-by-line
    if nfs == 0:
        X = np.vstack((X0, np.zeros((nfmax - 1, n))))
        F = np.zeros((nfmax, m))
        nf = 0  # in Matlab this is 1
        F0 = np.atleast_2d(fun(X[nf]))
        if F0.shape[1] != m:
            flag = -1
            log_error(flag)
            X, F = prepare_outputs_before_return(X, F, nf)
            return X, F, flag, xkin
        F[nf] = F0
        if np.any(np.isnan(F[nf])):
            flag = -3
            log_error(flag)
            X, F = prepare_outputs_before_return(X, F, nf)
            return X, F, flag, xkin
        log("%4i    Initial point  %11.5e" % (nf, np.sum(F[nf, :] ** 2)))
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
    ng = np.nan  # Needed for early termination, e.g., if a model is never built
    while nf + 1 < nfmax:
        #  1a. Compute the interpolation set.
        D = X[: nf + 1] - X[xkin]
        Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
        [Mdir, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xkin, npmax, Par, 0)
        if mp < n:
            [Mdir, mp] = bmpts(X[xkin], Mdir[0 : n - mp, :], L, U, delta, Par[2])
            for i in range(int(min(n - mp, nfmax - (nf + 1)))):
                nf += 1
                X[nf] = np.minimum(U, np.maximum(L, X[xkin] + Mdir[i, :]))
                F[nf] = fun(X[nf])
                if np.any(np.isnan(F[nf])):
                    flag = -3
                    log_error(flag)
                    X, F = prepare_outputs_before_return(X, F, nf)
                    return X, F, flag, xkin
                Fs[nf] = hfun(F[nf])
                log("%4i   Geometry point  %11.5e" % (nf, Fs[nf]))
                D = Mdir[i, :]
                Res[nf, :] = (F[nf, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
            if nf + 1 >= nfmax:
                break
            [_, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xkin, npmax, Par, False)
            if mp < n:
                flag = -5
                log_error(flag)
                X, F = prepare_outputs_before_return(X, F, nf)
                return

        #  1b. Update the quadratic model
        Cres = F[xkin]
        Hres = Hres + Hresdel
        c = Fs[xkin]
        G, H = combinemodels(Cres, Gres, Hres)
        ind_Lnotbinding = (X[xkin] > L) * (G.T > 0)
        ind_Unotbinding = (X[xkin] < U) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lnotbinding + ind_Unotbinding).T, 2)

        IERR = np.zeros(len(Mind))
        for i in range(len(Mind)):
            D = X[Mind[i]] - X[xkin]
            IERR[i] = (c - Fs[Mind[i]]) + [D @ (G + 0.5 * H @ D)]
        if np.any(Fs[Mind] == 0.0):
            ierror = np.nan
        else:
            ierror = np.linalg.norm(IERR / np.abs(Fs[Mind]), np.inf)
        log(progstr % (nf, delta, valid, mp, Fs[xkin], ng, ierror))
        jerr = np.zeros((len(Mind), m))
        for i in range(len(Mind)):
            D = X[Mind[i]] - X[xkin]
            for j in range(m):
                jerr[i, j] = (Cres[j] - F[Mind[i], j]) + D @ (Gres[:, j] + 0.5 * Hres[:, :, j] @ D)
        log_debug(jerr, 0)
        # input("Enter a key and press Enter to continue\n") - Don't uncomment when using Pytest with test_pounders.py

        # 2. Critically test invoked if the projected model gradient is small
        if ng < gtol:
            delta = max(gtol, np.max(np.abs(X[xkin])) * eps)
            [Mdir, _, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xkin, npmax, Par, 1)
            if not valid:
                [Mdir, mp] = bmpts(X[xkin], Mdir, L, U, delta, Par[2])
                for i in range(min(n - mp, nfmax - (nf + 1))):
                    nf += 1
                    X[nf] = np.minimum(U, np.maximum(L, X[xkin] + Mdir[i, :]))
                    F[nf] = fun(X[nf])
                    if np.any(np.isnan(F[nf])):
                        flag = -3
                        log_error(flag)
                        X, F = prepare_outputs_before_return(X, F, nf)
                        return X, F, flag, xkin
                    Fs[nf] = hfun(F[nf])
                    log("%4i   Critical point  %11.5e" % (nf, Fs[nf]))
                if nf + 1 >= nfmax:
                    break
                # Recalculate gradient based on a MFN model
                [_, _, valid, Gres, Hres, Mind] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xkin, npmax, Par, 0)
                G, H = combinemodels(Cres, Gres, Hres)
                ind_Lnotbinding = (X[xkin] > L) * (G.T > 0)
                ind_Unotbinding = (X[xkin] < U) * (G.T < 0)
                ng = np.linalg.norm(G * (ind_Lnotbinding + ind_Unotbinding).T, 2)
            if ng < gtol:
                flag = 0
                X, F = prepare_outputs_before_return(X, F, nf)
                log("g is sufficiently small")
                return X, F, flag, xkin

        # 3. Solve the subproblem min{G.T * s + 0.5 * s.T * H * s : Lows <= s <= Upps }
        Lows = np.maximum(L - X[xkin], -delta * np.ones((np.shape(L))))
        Upps = np.minimum(U - X[xkin], delta * np.ones((np.shape(U))))
        if spsolver == 1:  # Stefan's crappy 10line solver
            [Xsp, mdec] = bqmin(H, G, Lows, Upps)
        elif spsolver == 2:  # Arnold Neumaier's minq5
            [Xsp, mdec, minq_err, _] = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
            if minq_err < 0:
                flag = -4
                log_error(flag)
                X, F = prepare_outputs_before_return(X, F, nf)
                return X, F, flag, xkin
        # elif spsolver == 3:  # Arnold Neumaier's minq8
        #     [Xsp, mdec, minq_err, _] = minq8(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        #     assert minq_err >= 0, "Input error in minq"
        Xsp = Xsp.squeeze()
        step_norm = np.linalg.norm(Xsp, np.inf)

        # 4. Evaluate the function at the new point
        if (step_norm >= 0.01 * delta or valid) and not (mdec == 0 and not valid):
            Xsp = np.minimum(U, np.maximum(L, X[xkin] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (U[i] - Xsp[i] < eps * abs(U[i])) and (U[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = U[i]
                    log("eps project!")
                elif (Xsp[i] - L[i] < eps * abs(L[i])) and (L[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = L[i]
                    log("eps project!")

            if mdec == 0 and valid and np.array_equiv(Xsp, X[xkin]):
                flag = -2
                log_error(flag)
                X, F = prepare_outputs_before_return(X, F, nf)
                return X, F, flag, xkin

            nf += 1
            X[nf] = Xsp
            F[nf] = fun(X[nf])
            if np.any(np.isnan(F[nf])):
                flag = -3
                log_error(flag)
                X, F = prepare_outputs_before_return(X, F, nf)
                return X, F, flag, xkin
            Fs[nf] = hfun(F[nf])

            if mdec != 0:
                rho = (Fs[nf] - Fs[xkin]) / mdec
            else:
                if Fs[nf] == Fs[xkin]:
                    flag = -2
                    log_error(flag)
                    X, F = prepare_outputs_before_return(X, F, nf)
                    return X, F, flag, xkin
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
            log_warning("skipping sp soln!-----------")
        # 5. Evaluate a model-improving point if necessary
        if not valid and (nf + 1 < nfmax) and (rho < eta1):  # Implies xkin, delta unchanged
            # Need to check because model may be valid after Xsp evaluation
            [Mdir, mp, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xkin, npmax, Par, 1)
            if not valid:  # ! One strategy for choosing model-improving point:
                # Update model (exists because delta & xkin unchanged)
                D = X[: nf + 1] - X[xkin]
                Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
                [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xkin, npmax, Par, False)
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
                    flag = -3
                    log_error(flag)
                    X, F = prepare_outputs_before_return(X, F, nf)
                    return X, F, flag, xkin
                Fs[nf] = hfun(F[nf])
                log("%4i   Model point     %11.5e" % (nf, Fs[nf]))
                if Fs[nf] < Fs[xkin]:  # ! Eventually check stuff decrease here
                    log("**improvement from model point****")
                    # Update model to reflect new base point
                    D = X[nf] - X[xkin]
                    xkin = nf  # Change current center
                    Cres = F[xkin]
                    # Don't actually use
                    for j in range(m):
                        Gres[:, j] = Gres[:, j] + Hres[:, :, j] @ D.T
    # Evaluation budget exceeded
    flag = ng
    log_error(flag)
    return X, F, flag, xkin
