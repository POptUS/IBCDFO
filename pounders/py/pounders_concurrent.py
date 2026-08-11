import numpy as np

from .create_trsp_solver import create_trsp_solver
from .bmpts import bmpts
from .checkinputss import checkinputss
from .formquad import formquad
from .prepare_outputs_before_return import prepare_outputs_before_return


def pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior=None, Options=None, Model=None):
    r"""
    This version of |pounders| calls ``Ffun`` with batches of model-building
    points.  In particular, when new geometry points are needed, |pounders|
    constructs the corresponding rows of ``X`` and calls

    .. code-block:: python

        Ffun(X[idx_new])

    rather than calling ``Ffun`` separately for each row of ``X``.

    Any parallelism/concurrency must be implemented inside ``Ffun`` rather than
    inside |pounders|.  Users who want concurrent evaluation of model-building
    points should provide an ``Ffun`` that accepts a batch of points
    ``X[idx_new]`` with shape ``(batch_size, n)``, where each row is one point
    to evaluate.  The user-provided ``Ffun`` should evaluate these
    ``batch_size`` points with whatever degree of concurrency is available and
    return an array with shape ``(batch_size, m)``, whose rows contain the
    corresponding ``Ffun`` outputs in the same order as the input points.

    Otherwise, this implementation and its interface are identical to those of
    the standard |pounders| implementation.  Please refer to
    :py:func:`ibcdfo.run_pounders` for more information.
    """
    # ----- HARDCODED VALUES
    BAD_ARGS_RETURN = ([], [], [], -1, -1)

    # ----- RENAME & SANITIZE GIVEN ARGUMENTS
    # Perform this first so that the renamed variables are available for
    # determining default values and error checking.
    delta = delta_0

    # For arguments that are specified as X-element Numpy arrays, we can be
    # flexible and accept any iterables that can be converted to genuinely 1D
    # arrays of the correct length.  We convert here into the final
    # specification required by the algorithm's implementation.
    X_0 = np.atleast_1d(np.squeeze(X_0))
    Low = np.atleast_1d(np.squeeze(Low))
    Upp = np.atleast_1d(np.squeeze(Upp))

    # ----- EXTRACT ARGUMENTS & DEFINE DEFAULTS
    # Once the different fields in dictionary arguments are extracted into local
    # variables, the dictionary arguments should no longer be used in favor of
    # the local arguments, which are carefully checked.
    #
    # Note that some arguments are used to compute default values before the
    # arguments are error checked.

    # -- Options dictionary
    # TODO: Many of these need to be added to the inline docs above.
    ALL_OPTIONS_KEYS = {"printf", "spsolver", "delta_max", "delta_min", "delta_inact", "gamma_dec", "gamma_inc", "eta_1", "hfun", "combinemodels"}
    if Options is None:
        Options = {}
    if not set(Options).issubset(ALL_OPTIONS_KEYS):
        extras = set(Options).difference(ALL_OPTIONS_KEYS)
        print(f"Error: Options dictionary contains unknown keys {extras}")
        return BAD_ARGS_RETURN

    printf = Options.get("printf", 0)
    spsolver = Options.get("spsolver", 2)
    delta_max = Options.get("delta_max", np.minimum(0.5 * np.min(Upp - Low), (10**3) * delta))
    delta_min = Options.get("delta_min", np.minimum(delta * (10**-13), g_tol / 10))
    delta_inact = Options.get("delta_inact", 0.75)
    gamma_dec = Options.get("gamma_dec", 0.5)
    gamma_inc = Options.get("gamma_inc", 2)
    eta_1 = Options.get("eta1", 0.05)

    if "hfun" in Options:
        hfun = Options["hfun"]
        combinemodels = Options["combinemodels"]
    else:
        from .general_h_funs import h_leastsquares as hfun
        from .general_h_funs import combine_leastsquares as combinemodels

    solve_trsp = create_trsp_solver(spsolver)

    # -- Model dictionary
    ALL_MODEL_KEYS = {"np_max", "Par"}
    DEFAULT_MODEL_PAR = [np.sqrt(n), np.maximum(10, np.sqrt(n)), 10**-3, 0.001, 0]
    if Model is None:
        Model = {}
    if not set(Model).issubset(ALL_MODEL_KEYS):
        extras = set(Model).difference(ALL_MODEL_KEYS)
        print(f"Error: Model dictionary contains unknown keys {extras}")
        return BAD_ARGS_RETURN

    np_max = Model.get("np_max", 2 * n + 1)
    Par = Model.get("Par", DEFAULT_MODEL_PAR)

    # -- Prior dictionary
    EXPECTED_PRIOR_KEYS = {"nfs", "X_init", "F_init", "xk_in"}
    if Prior is None:
        Prior = {"nfs": 0, "X_init": np.full((0, n), np.nan, float), "F_init": np.full((0, m), np.nan, float), "xk_in": 0}
    if set(Prior) != EXPECTED_PRIOR_KEYS:
        print(f"Error: Prior must be a dictionary with the keys {EXPECTED_PRIOR_KEYS}")
        return BAD_ARGS_RETURN

    nfs = Prior["nfs"]
    X_init = Prior["X_init"]
    F_init = Prior["F_init"]
    xk_in = Prior["xk_in"]

    # -- Strict error checking of local variables based on what the implementation requires
    # This does not alter any of the local arguments.
    flag = checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta_0, nfs, m, X_init, F_init, xk_in, Low, Upp)
    if flag == -1:
        return [], [], [], flag, xk_in

    # ----- OPTIMIZE!
    eps = np.finfo(float).eps  # Define machine epsilon
    if printf:
        print("  nf   delta    fl  np       f0           g0       ierror")
        progstr = "%4i %9.2e %2i %3i  %11.5e %12.4e %11.3e\n"  # Line-by-line

    if nfs == 0:
        X = np.vstack((X_0, np.zeros((nf_max - 1, n))))
        F = np.zeros((nf_max, m))
        hF = np.zeros(nf_max)
        nf = 0  # in Matlab this is 1
        F_0 = np.atleast_2d(Ffun(X[nf]))
        if F_0.shape[1] != m:
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -1)
            return X, F, hF, flag, xk_in
        F[nf] = F_0
        if np.any(np.isnan(F[nf])) or np.any(np.isinf(F[nf])):
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
            return X, F, hF, flag, xk_in
        if printf:
            print("%4i    Initial point  %11.5e\n" % (nf, hfun(F[nf, :])))
    else:
        X = np.vstack((X_init, np.zeros((nf_max, n))))
        F = np.vstack((F_init, np.zeros((nf_max, m))))
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
        #  1a. Compute the interpolation set.
        D = X[: nf + 1] - X[xk_in]
        Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
        [Mdir, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, np_max, Par, False)
        if mp < n:
            [Mdir, mp] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Par[2])
            k_new = int(min(n - mp, nf_max - (nf + 1)))  # new geometry points to send to Ffun (while respecting nfmax)
            idx_new = nf + 1 + np.arange(k_new)  # absolute indices of these points

            X[idx_new] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[:k_new, :]))
            F[idx_new] = Ffun(X[idx_new])

            if np.any(np.isnan(F[idx_new])) or np.any(np.isinf(F[idx_new])):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                return X, F, hF, flag, xk_in

            for i in range(k_new):
                nf += 1
                hF[nf] = hfun(F[nf])
                if printf:
                    print("%4i   Geometry point  %11.5e\n" % (nf, hF[nf]))
                D = Mdir[i, :]
                Res[nf, :] = (F[nf, :] - Cres) - 0.5 * D @ np.tensordot(D.T, Hres, 1)
            if nf + 1 >= nf_max:
                break
            [_, mp, valid, Gres, Hresdel, Mind] = formquad(X[0 : nf + 1, :], Res[0 : nf + 1, :], delta, xk_in, np_max, Par, False)
            if mp < n:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -5)
                return X, F, hF, flag, xk_in

        #  1b. Update the quadratic model
        Cres = F[xk_in]
        Hres = Hres + Hresdel
        c = hF[xk_in]
        G, H = combinemodels(Cres, Gres, Hres)
        ind_Lownotbinding = (X[xk_in] > Low) * (G.T > 0)
        ind_Uppnotbinding = (X[xk_in] < Upp) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lownotbinding + ind_Uppnotbinding).T, 2)
        if printf:
            IERR = np.zeros(len(Mind))
            for i in range(len(Mind)):
                D = X[Mind[i]] - X[xk_in]
                IERR[i] = (c - hF[Mind[i]]) + (D @ (G + 0.5 * H @ D))
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
            delta = np.maximum(g_tol, np.max(np.abs(X[xk_in])) * eps)
            [Mdir, _, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, np_max, Par, True)
            if not valid:
                [Mdir, mp] = bmpts(X[xk_in], Mdir, Low, Upp, delta, Par[2])
                for i in range(min(n - mp, nf_max - (nf + 1))):
                    nf += 1
                    X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[i, :]))
                    F[nf] = Ffun(X[nf])
                    if np.any(np.isnan(F[nf])) or np.any(np.isinf(F[nf])):
                        X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                        return X, F, hF, flag, xk_in
                    hF[nf] = hfun(F[nf])
                    if printf:
                        print("%4i   Critical point  %11.5e\n" % (nf, hF[nf]))
                if nf + 1 >= nf_max:
                    break
                # Recalculate gradient based on a MFN model
                [_, _, valid, Gres, Hres, Mind] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, np_max, Par, False)
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
        [Xsp, mdec, found_solution] = solve_trsp(H, G, Lows, Upps)
        if not found_solution:
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -4)
            return X, F, hF, flag, xk_in

        step_norm = np.linalg.norm(Xsp, np.inf) if n > 1 else np.abs(Xsp)

        # 4. Evaluate the function at the new point (provided the model is
        # valid, or the step is sufficiently large and mdec isn't zero)
        if valid or (step_norm >= 0.01 * delta and mdec != 0):
            Xsp = np.minimum(Upp, np.maximum(Low, X[xk_in] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (Upp[i] - Xsp[i] < eps * abs(Upp[i])) and (Upp[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = Upp[i]
                    print("eps project!")
                elif (Xsp[i] - Low[i] < eps * abs(Low[i])) and (Low[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = Low[i]
                    print("eps project!")

            if mdec == 0 and valid and np.array_equiv(Xsp, X[xk_in]) and delta < np.sqrt(eps):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                return X, F, hF, flag, xk_in

            nf += 1
            X[nf] = Xsp
            if np.array_equiv(Xsp, X[xk_in]):
                # We don't want to do the expensive F eval if Xsp is already in X
                F[nf] = F[xk_in]
            else:
                F[nf] = Ffun(X[nf])

            if np.any(np.isnan(F[nf])) or np.any(np.isinf(F[nf])):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                return X, F, hF, flag, xk_in
            hF[nf] = hfun(F[nf])

            if mdec != 0:
                rho = (hF[nf] - hF[xk_in]) / mdec
            else:  # Note: this conditional only occurs when model is valid
                if hF[nf] == hF[xk_in]:
                    if delta < np.sqrt(eps):
                        X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                        return X, F, hF, flag, xk_in
                    else:
                        rho = -np.inf
                else:
                    rho = np.inf * np.sign(hF[nf] - hF[xk_in])

            # 4a. Update the center
            if (rho >= eta_1) or (rho > 0 and valid):
                # Update model to reflect new center
                Cres = F[xk_in]
                xk_in = nf  # Change current center
            # 4b. Update the trust-region radius:
            if (rho >= eta_1) and (step_norm > delta_inact * delta):
                delta = np.minimum(delta * gamma_inc, delta_max)
            elif valid:
                delta = delta * gamma_dec
                if delta <= delta_min:
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -6)
                    return X, F, hF, flag, xk_in

        else:  # Don't evaluate f at Xsp
            rho = -1  # Force yourself to do a model-improving point
            if printf:
                print("Warning: skipping sp soln!-----------")
        # 5. Evaluate a model-improving point if necessary
        if not valid and (nf + 1 < nf_max) and (rho < eta_1):  # Implies xk_in, delta unchanged
            # Need to check because model may be valid after Xsp evaluation
            [Mdir, mp, valid, _, _, _] = formquad(X[: nf + 1, :], F[: nf + 1, :], delta, xk_in, np_max, Par, True)
            if not valid:  # ! One strategy for choosing model-improving point:
                # Update model (exists because delta & xk_in unchanged)
                D = X[: nf + 1] - X[xk_in]
                Res[: nf + 1, :] = (F[: nf + 1, :] - Cres) - np.diagonal(0.5 * D @ (np.tensordot(D, Hres, axes=1))).T
                [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xk_in, np_max, Par, False)
                if len(Mind) < n + 1:
                    # This is almost never triggered but is a safeguard for
                    # pathological cases where one needs to recover from
                    # unusual conditioning of recent interpolation sets
                    Par[4] = 1
                    [_, _, valid, Gres, Hresdel, Mind] = formquad(X[: nf + 1, :], Res[: nf + 1, :], delta, xk_in, np_max, Par, False)
                    Par[4] = 0
                Hres = Hres + Hresdel
                # Update for modelimp; Cres unchanged b/c xk_in unchanged
                G, H = combinemodels(Cres, Gres, Hres)
                # Evaluate model-improving points to pick best one
                # May eventually want to normalize Mdir first for infty norm
                # Plus directions
                [Mdir1, mp1] = bmpts(X[xk_in], Mdir[0 : n - mp, :], Low, Upp, delta, Par[2])
                for i in range(n - mp1):
                    D = Mdir1[i, :]
                    Res[i, 0] = D @ (G + 0.5 * H @ D.T)
                b = np.argmin(Res[: n - mp1, 0:1])
                a1 = np.min(Res[: n - mp1, 0:1])
                Xsp = Mdir1[b, :]
                # Minus directions
                [Mdir1, mp2] = bmpts(X[xk_in], -Mdir[0 : n - mp, :], Low, Upp, delta, Par[2])
                for i in range(n - mp2):
                    D = Mdir1[i, :]
                    Res[i, 0] = D @ (G + 0.5 * H @ D.T)
                b = np.argmin(Res[: n - mp2, 0:1])
                a2 = np.min(Res[: n - mp2, 0:1])
                if a2 < a1:
                    Xsp = Mdir1[b, :]
                nf += 1
                X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Xsp))  # Temp safeguard
                F[nf] = Ffun(X[nf])
                if np.any(np.isnan(F[nf])) or np.any(np.isinf(F[nf])):
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                    return X, F, hF, flag, xk_in
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
                    for j in range(m):
                        Gres[:, j] = Gres[:, j] + Hres[:, :, j] @ D.T
    if printf:
        print("Number of function evals exceeded")
    flag = ng
    return X, F, hF, flag, xk_in
