import sys

import numpy as np

from .bmpts import bmpts
from .bqmin import bqmin
from .checkinputss import checkinputss
from .formquad import formquad, formquad_model_improvement
from .prepare_outputs_before_return import prepare_outputs_before_return


def _default_model_par_values(n):
    par = np.zeros(4)
    par[0] = np.sqrt(n)
    par[1] = max(10, np.sqrt(n))
    par[2] = 10**-3
    par[3] = 0.001

    return par


def _default_model_np_max(n):
    return 2 * n + 1


def _default_prior():
    Prior = {}
    Prior["nfs"] = 0
    Prior["X_init"] = []
    Prior["F_init"] = []
    Prior["xk_in"] = 0

    return Prior


def pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior=None, Options=None, Model=None):
    """
    POUNDERS: Practical Optimization Using No Derivatives for sums of Squares
      [X, F, hF, flag, xk_in] = pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp)

    This code minimizes output from a structured blackbox function, solving
    min { f(X)=sum_(i=1:m) F_i(x)^2, such that Low_j <= X_j <= Upp_j, j=1,...,n }
    where the user-provided blackbox F is specified in the handle Ffun. Evaluation
    of this F must result in the return of a 1-by-m row vector. Bounds must be
    specified in Upp and Low but can be set to Low=-Inf(1,n) and Upp=Inf(1,n) if the
    unconstrained solution is desired. The algorithm will not evaluate F
    outside of these bounds, but it is possible to take advantage of function
    values at infeasible X if these are passed initially through (X_init, F_init).
    In each iteration, the algorithm forms an interpolating quadratic model
    of the function and minimizes it in an infinity-norm trust region.

    Optionally, a user can specify an outer function (hfun) that maps the
    elements of F to a scalar value (to be minimized). Doing this also requires
    a function handle (combinemodels) that tells pounders how to map the linear
    and quadratic terms from the models of the F_i into a single quadratic TRSP
    model.

    This software comes with no warranty, is not bug-free, and is not for
    industrial use or public distribution.
    Direct requests and bugs to wild@mcs.anl.gov.
    A technical report/manual is forthcoming, a brief description is in
    Nuclear Energy Density Optimization. Phys. Rev. C, 82:024313, 2010.

    --INPUTS-----------------------------------------------------------------
    Ffun    [f h] Function handle so that Ffun(x) evaluates F (@calfun)
    X_0     [dbl] [1-by-n] Initial point (zeros(1,n))
    n       [int] Dimension (number of continuous variables)
    nf_max  [int] Maximum number of function evaluations (>=n+1) (100)
    g_tol   [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
    delta_0 [dbl] Positive initial trust region radius (.1)
    m       [int] Number of components returned from Ffun
    Low     [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
    Upp     [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))

    Prior   [dict] of past evaluations of values Ffun with keys:
        X_init  [dbl] [nfs-by-n] Set of initial points
        F_init  [dbl] [nfs-by-m] Set of values for points in X_init
        xk_in   [int] Index in X_init for initial starting point
        nfs     [int] Number of function values in F_init known in advance

    Options [dict] of options to the method
        printf   [int] 0 No printing to screen (default)
                       1 Debugging level of output to screen
                       2 More verbose screen output
        spsolver       [int] Trust-region subproblem solver flag (2)
        hfun           [f h] Function handle for mapping output from F
        combinemodels  [f h] Function handle for combining models of F

    Model   [dict] of options for model building
        np_max  [int] Maximum number of interpolation points (>=n+1) (2*n+1)
        Par     [1-by-4] list for formquad

    --OUTPUTS----------------------------------------------------------------
    X       [dbl] [nf_max+nfs-by-n] Locations of evaluated points
    F       [dbl] [nf_max+nfs-by-m] Ffun values of evaluated points in X
    hF      [dbl] [nf_max+nfs-by-1] Composed values h(Ffun) for evaluated points in X
    flag    [dbl] Termination criteria flag:
                  = 0 normal termination because of grad,
                  > 0 exceeded nf_max evals,   flag = norm of grad at final X
                  = -1 if input was fatally incorrect (error message shown)
                  = -2 if a valid model produced X[nf] == X[xk_in] or (mdec == 0, hF[nf] == hF[xk_in])
                  = -3 error if a NaN was encountered
                  = -4 error in TRSP Solver
                  = -5 unable to get model improvement with current parameters
    xk_in    [int] Index of point in X representing approximate minimizer
    """
    if Options is None:
        Options = {}

    if Model is None:
        Model = {}
        Model["H_flag"] = True
        Model["Par"] = _default_model_par_values(n)
        Model["np_max"] = _default_model_np_max(n)
    else:
        if "H_flag" not in Model: 
            Model["H_flag"] = True
        if "Par" not in Model:
            Model["Par"] = _default_model_par_values(n)
        if "np_max" not in Model:
            Model["np_max"] = _default_model_np_max(n)

    if Prior is None:
        Prior = _default_prior()
    else:
        key_list = ["nfs", "X_init", "F_init", "xk_in"]
        assert set(Prior.keys()) == set(key_list), f"Prior keys must be {key_list}"
        Prior["X_init"] = np.atleast_2d(Prior["X_init"])
        if Prior["X_init"].ndim == 2 and Prior["X_init"].shape[1] == 1:
            Prior["X_init"] = Prior["X_init"].T

    nfs = Prior["nfs"]
    delta = delta_0
    spsolver = Options.get("spsolver", 2)
    delta_max = Options.get("delta_max", min(0.5 * np.min(Upp - Low), (10**3) * delta))
    delta_min = Options.get("delta_min", min(delta * (10**-13), g_tol / 10))
    gamma_dec = Options.get("gamma_dec", 0.5)
    gamma_inc = Options.get("gamma_inc", 2)
    eta_1 = Options.get("eta1", 0.05)
    printf = Options.get("printf", 0)
    delta_inact = Options.get("delta_inact", 0.75)

    if "hfun" in Options:
        hfun = Options["hfun"]
        combinemodels = Options["combinemodels"]
    else:
        hfun = lambda F: np.sum(F**2)
        from .general_h_funs import leastsquares as combinemodels

    # choose your spsolver
    if spsolver == 2:
        try:
            from minqsw import minqsw
        except ModuleNotFoundError as e:
            print(e)
            sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

    [flag, X_0, _, F_init, Low, Upp, xk_in] = checkinputss(Ffun, X_0, n, Model["np_max"], nf_max, g_tol, delta_0, Prior["nfs"], m, Prior["X_init"], Prior["F_init"], Prior["xk_in"], Low, Upp)
    if flag == -1:
        X = []
        F = []
        hF = []
        return X, F, hF, flag, xk_in
    eps = np.finfo(float).eps  # Define machine epsilon
    if printf:
        print("  nf   delta    fl  np       f0           g0       ierror")
        progstr = "%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n"  # Line-by-line
    if Prior["nfs"] == 0:
        X = np.vstack((X_0, np.zeros((nf_max - 1, n))))
        F = np.zeros((nf_max, m))
        hF = np.zeros(nf_max)
        nf = 0  # in Matlab this is 1
        F_0 = np.atleast_2d(Ffun(X[nf]))
        if F_0.shape[1] != m:
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -1)
            return X, F, hF, flag, xk_in
        F[nf] = F_0
        if np.any(np.isnan(F[nf])):
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
            return X, F, hF, flag, xk_in
        if printf:
            print("%4i    Initial point  %11.5e\n" % (nf, hfun(F[nf, :])))
    else:
        X = np.vstack((Prior["X_init"], np.zeros((nf_max, n))))
        F = np.vstack((Prior["F_init"], np.zeros((nf_max, m))))
        hF = np.zeros(nf_max + nfs)
        nf = nfs - 1
        nf_max = nf_max + nfs
    for i in range(nf + 1):
        hF[i] = hfun(F[i])
    Res = np.zeros(np.shape(F))
    Cres = F[xk_in]
    Hres = np.zeros((n, n, m))
    ng = np.nan  # Needed for early termination, e.g., if a model is never built
    while nf + 1 < nf_max:
        #  1a. Compute displacements
        D = X[: nf + 1] - X[xk_in]
        Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T

        # Request directions for points to be evaluated
        [Mdir, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False, Model["H_flag"], Hres)

        if mp < n:
            [Mdir, mp] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Model["Par"][2])
            for i in range(int(min(n - mp, nf_max - (nf + 1)))):
                nf += 1
                X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[i, :]))
                F[nf] = Ffun(X[nf])
                if np.any(np.isnan(F[nf])):
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                    return X, F, hF, flag, xk_in
                hF[nf] = hfun(F[nf])
                if printf:
                    print("%4i   Geometry point  %11.5e\n" % (nf, hF[nf]))
                D = Mdir[i, :]
                Res[nf, :] = (F[nf, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
            if nf + 1 >= nf_max:
                break

            [_, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False, Model["H_flag"], Hres)

        c = hF[xk_in]
        G, H = combinemodels(Cres, Gres, Hres)
        ind_Lownotbinding = (X[xk_in] > Low) * (G.T > 0)
        ind_Uppnotbinding = (X[xk_in] < Upp) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lownotbinding + ind_Uppnotbinding).T, 2)
        if printf:
            IERR = np.zeros(len(Mind))
            for i in range(len(Mind)):
                D = X[Mind[i]] - X[xk_in]
                IERR[i] = (c - hF[Mind[i]]) + [D @ (G + 0.5 * H @ D)]
            if np.any(hF[Mind] == 0.0):
                ierror = np.nan
            else:
                ierror = np.linalg.norm(IERR / np.abs(hF[Mind]), np.inf)
            print(progstr % (nf, delta, valid, mp, hF[xk_in], ng, ierror))
            if printf >= 2:
                jerr = np.zeros((len(Mind), m))
                for i in range(len(Mind)):
                    D = X[Mind[i]] - X[xk_in]
                    for j in range(m):
                        jerr[i, j] = (Cres[j] - F[Mind[i], j]) + D @ (Gres[:, j] + 0.5 * Hres[:, :, j] @ D)
                print(jerr)
            # input("Enter a key and press Enter to continue\n") - Don't uncomment when using Pytest with test_pounders.py
        # 2. Critically test invoked if the projected model gradient is small
        if ng < g_tol:
            delta = max(g_tol, np.max(np.abs(X[xk_in])) * eps)
            [Mdir, _, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], 1)
            if not valid:
                [Mdir, mp] = bmpts(X[xk_in], Mdir, Low, Upp, delta, Model["Par"][2])
                for i in range(min(n - mp, nf_max - (nf + 1))):
                    nf += 1
                    X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[i, :]))
                    F[nf] = Ffun(X[nf])
                    if np.any(np.isnan(F[nf])):
                        X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                        return X, F, flag, xk_in
                    hF[nf] = hfun(F[nf])
                    if printf:
                        print("%4i   Critical point  %11.5e\n" % (nf, hF[nf]))
                if nf + 1 >= nf_max:
                    break
                # Recalculate gradient based on a MFN model
                [_, _, valid, Gres, Hres, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], 0)
                G, H = combinemodels(Cres, Gres, Hres)
                ind_Lownotbinding = (X[xk_in] > Low) * (G.T > 0)
                ind_Uppnotbinding = (X[xk_in] < Upp) * (G.T < 0)
                ng = np.linalg.norm(G * (ind_Lownotbinding + ind_Uppnotbinding).T, 2)
            if ng < g_tol:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, 0)
                return X, F, hF, flag, xk_in

        # 3. Solve the subproblem min{G.T * s + 0.5 * s.T * H * s : Lows <= s <= Upps }
        Lows = np.maximum(Low - X[xk_in], -delta * np.ones((np.shape(Low))))
        Upps = np.minimum(Upp - X[xk_in], delta * np.ones((np.shape(Upp))))
        if spsolver == 1:  # Stefan's crappy 10line solver
            [Xsp, mdec] = bqmin(H, G, Lows, Upps)
        elif spsolver == 2:  # Arnold Neumaier's minq5
            [Xsp, mdec, minq_err, _] = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
            if minq_err < 0:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -4)
                return X, F, hF, flag, xk_in
        # elif spsolver == 3:  # Arnold Neumaier's minq8
        #     [Xsp, mdec, minq_err, _] = minq8(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        #     assert minq_err >= 0, "Input error in minq"
        Xsp = Xsp.squeeze()
        step_norm = np.linalg.norm(Xsp, np.inf)

        # 4. Evaluate the function at the new point
        if (step_norm >= 0.01 * delta or valid) and not (mdec == 0 and not valid):
            Xsp = np.minimum(Upp, np.maximum(Low, X[xk_in] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (Upp[i] - Xsp[i] < eps * abs(Upp[i])) and (Upp[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = Upp[i]
                    print("eps project!")
                elif (Xsp[i] - Low[i] < eps * abs(Low[i])) and (Low[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = Low[i]
                    print("eps project!")

            if mdec == 0 and valid and np.array_equiv(Xsp, X[xk_in]):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                return X, F, hF, flag, xk_in

            nf += 1
            X[nf] = Xsp
            F[nf] = Ffun(X[nf])
            if np.any(np.isnan(F[nf])):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                return X, F, hF, flag, xk_in
            hF[nf] = hfun(F[nf])

            if mdec != 0:
                rho = (hF[nf] - hF[xk_in]) / mdec
            else:
                if hF[nf] == hF[xk_in]:
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                    return X, F, hF, flag, xk_in
                else:
                    rho = np.inf * np.sign(hF[nf] - hF[xk_in])

            # 4a. Update the center
            if (rho >= eta_1) or (rho > 0 and valid):
                # Update model to reflect new center
                Cres = F[xk_in]
                xk_in = nf  # Change current center
            # 4b. Update the trust-region radius:
            if (rho >= eta_1) and (step_norm > delta_inact * delta):
                delta = min(delta * gamma_inc, delta_max)
            elif valid:
                delta = max(delta * gamma_dec, delta_min)
        else:  # Don't evaluate f at Xsp
            rho = -1  # Force yourself to do a model-improving point
            if printf:
                print("Warning: skipping sp soln!-----------")
        # 5. Evaluate a model-improving point if necessary
        if not valid and (nf + 1 < nf_max) and (rho < eta_1):  # Implies xk_in, delta unchanged
            # Need to check because model may be valid after Xsp evaluation
            [Mdir, mp, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], True)

            if not valid:  # ! One strategy for choosing model-improving point:
                # Update model (exists because delta & xk_in unchanged)
                D = X[: nf + 1] - X[xk_in]
                Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T

                [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xk_in, Model["np_max"], Model["Par"], False, Model["H_flag"], Hres)

                Xsp = formquad_model_improvement(X[xk_in], Cres, Gres, Hres, Mdir, mp, Low, Upp, delta, Model, combinemodels)

                nf += 1
                X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Xsp))  # Temp safeguard
                F[nf] = Ffun(X[nf])
                if np.any(np.isnan(F[nf])):
                    raise NanValueError("NaN encountered")
                hF[nf] = hfun(F[nf])
                if printf:
                    print("%4i   Model point     %11.5e\n" % (nf, hF[nf]))
                if hF[nf] < hF[xk_in]:  # ! Eventually check stuff decrease here
                    if printf:
                        print("**improvement from model point****")
                    # Update model to reflect new base point
                    D = X[nf] - X[xk_in]
                    xk_in = nf  # Change current center
                    Cres = F[xk_in]
                    # Don't actually use
                    # for j in range(m):
                    #     Gres[:, j] = Gres[:, j] + Hres[:, :, j] @ D.T

    if printf:
        print("Number of function evals exceeded")
    flag = ng
    return X, F, hF, flag, xk_in
