import numbers

import numpy as np

from .defaults import (
    ALL_MODEL_KEYS,
    ALL_OPTIONS_KEYS,
    EXPECTED_PRIOR_KEYS,
    compute_default_prior,
    compute_default_model,
    compute_default_options,
)
from .create_trsp_solver import create_trsp_solver
from .bmpts import bmpts
from .checkinputss import checkinputss
from .formquad import formquad
from .prepare_outputs_before_return import prepare_outputs_before_return


def pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior=None, Options=None, Model=None):
    r"""
    Run |pounders| on the optimization problem specified by the given
    arguments.

    :param Ffun:    Function that returns :math:`\Ffun(\psp)` as
        :math:`\nd`-element NumPy array for given :math:`\psp`
    :param X_0:     :math:`\np`-element 1D NumPy array that specifies the
        initial point, which must satisfy the boundary constraints
    :param n:       Dimension (number of continuous, real-valued input variables)
    :param nf_max:  Maximum number of function evaluations (:math:`> \np+1` if
        **Prior** not provided or **nfs** = 0; :math:`\ge \np+1`, if **nfs** > 0)
    :param g_tol:   Tolerance for the 2-norm of the model gradient
    :param delta_0: Positive initial trust region radius
    :param m:       Dimension of output of ``Ffun`` (number of component functions)
    :param Low:     :math:`\np`-element 1D NumPy array of lower bounds
    :param Upp:     :math:`\np`-element 1D NumPy array of upper bounds
    :param Prior:   ``dict`` describing past evaluations of ``Ffun``.  Set to ``None`` to run optimization assuming no past evaluations. A nonempty **Prior** must contain entries:

        * **nfs** - Number of past function evaluations
        * **X_init** - :math:`\mathrm{nfs} \times \np` NumPy array of distinct
          points :math:`\psp_k`
        * **F_init** - :math:`\mathrm{nfs} \times \nd` NumPy array of values
          :math:`\Ffun(\psp_k)` computed with ``Ffun``
        * **xk_in** - Zero-based index into ``X_init`` and ``F_init`` that
          corresponds to the point and value to use as the initial point for
          optimization. Note that if **Prior** is nonempty, **X_init[xk_in]**
          and **X_0** must be identical and still satisfy the boundary
          constraints.

    :param Options: ``dict`` of method options.  Set to ``None`` to use default
        values.

        * **printf** (default is 0)

            * 0 - No printing to screen
            * 1 - Debugging level of output to screen
            * 2 - More verbose screen output

        * **spsolver** - Trust-region subproblem solver flag

            * ``ibcdfo.pounders.TRSP_SOLVER_MINQ5`` - Arnold Neumaier's minq5 solver (default)

        * **delta_max** - Maximum allowed trust-region radius (default is
          :math:`\min(\min(\mathrm{Upp}-\mathrm{Low})/2, 10^3\cdot\mathrm{delta\_0})`)
        * **delta_min** - Minimum allowed trust-region radius; falling at or
          below this triggers termination (default is
          :math:`\min(10^{-13}\cdot\mathrm{delta\_0}, \mathrm{g\_tol}/10)`)
        * **delta_inact** - Fraction of the trust-region radius the step norm
          must exceed for a successful step to also grow the radius (default is 0.75)
        * **gamma_dec** - Factor by which the trust-region radius is
          decreased on an unsuccessful step (default is 0.5)
        * **gamma_inc** - Factor by which the trust-region radius is
          increased on a successful, sufficiently long step (default is 2)
        * **eta1** - Minimum ratio of actual to predicted reduction required
          to accept a step and update the center (default is 0.05)
        * **hfun** - Outer function :math:`\hfun` that maps given
          :math:`\Ffun(\psp)` to scalars for minimization (default
          :py:func:`ibcdfo.pounders.h_leastsquares` yields ordinary least
          squares)
        * **combinemodels** - Function that maps the linear and quadratic terms
          from the models of :math:`\Ffun` into a single quadratic model
          (default is ``ibcdfo.pounders.combine_leastsquares`` to match
          the default **hfun**)

    :param Model: ``dict`` of model building options.  Set to ``None`` to use
        default values.

        * **np_max** - Integer in :math:`\Z[\np+1, (n+1)(n+2)/2]` that specifies
          the maximum number of interpolation points (default is :math:`2\np+1`)
        * **Par** - Five-element ``list`` for ``formquad`` (default :math:`[\sqrt{n}, \max\{10,\sqrt{n}\}, 10^{-3}, 10^{-3}, 0]`)

    :return:
        * **X** - :math:`k \times \np` NumPy array containing locations of
          evaluated points (including those provided in ``Prior``) in the order
          in which they were evaluated, where
          :math:`\mathrm{nfs} < k \le \mathrm{nf\_max+nfs}`.
        * **F** - :math:`k \times \nd` NumPy array containing the function
          values at ``X`` with matching ordering.
        * **hF** - :math:`k`-element 1D NumPy array of composed values
          ``hfun(Ffun(x))`` at ``X`` with matching ordering.
        * **flag** - Termination criteria flag (see general |pounders| documentation)
        * **xk_in** - Zero-based index of point in ``X`` representing the
          incumbent at termination (approximate local minimizer if ``flag=0``)
    """
    # ----- UPFRONT ERROR CHECKING
    # These are used to set defaults before official error checking
    if not isinstance(n, numbers.Integral):
        raise TypeError(f"Error: dimension n is not an integer ({n})")
    if n < 1:
        raise ValueError(f"Error: dimension n is not a positive integer ({n})")

    if not isinstance(m, numbers.Integral):
        raise TypeError(f"Error: dimension m is not an integer ({m})")
    if m < 1:
        raise ValueError(f"Error: dimension m is not a positive integer ({m})")

    # ----- ALLOW "1D" NUMPY ARRAY FLEXIBILITY
    # For arguments that are specified as X-element NumPy arrays, we can be
    # flexible and accept any iterables that can be converted to genuinely 1D
    # arrays of the correct length.  We eagerly convert here into the final
    # specification required by the algorithm's implementation.
    # TODO: Uncomment this once we add in tests to confirm this.  Ensure that we
    # test n=1/m=1 case as well.  Update docstrings as well to de-emphasize 1D.
    # X_0 = np.atleast_1d(np.squeeze(X_0))
    # Low = np.atleast_1d(np.squeeze(Low))
    # Upp = np.atleast_1d(np.squeeze(Upp))

    # ----- EXTRACT ARGUMENTS & DEFINE DEFAULTS
    # Once the different fields in dictionary arguments are extracted into local
    # variables, the dictionary arguments should no longer be used in favor of
    # the local arguments, which are carefully checked.

    # -- Model dictionary
    if Model is None:
        Model = {}
    if not isinstance(Model, dict):
        raise TypeError("Error: Model argument must be a dictionary")
    if not set(Model).issubset(ALL_MODEL_KEYS):
        extras = set(Model).difference(ALL_MODEL_KEYS)
        raise ValueError(f"Error: Model dictionary contains unknown keys {extras}")

    defaults = compute_default_model(n)
    for key, value in Model.items():
        defaults[key] = value

    np_max = defaults["np_max"]
    Par = defaults["Par"]

    # -- Prior dictionary
    if Prior is None:
        Prior = compute_default_prior(n, m)
    if not isinstance(Prior, dict):
        raise TypeError("Error: Prior argument must be a dictionary")
    if set(Prior) != EXPECTED_PRIOR_KEYS:
        raise ValueError(f"Error: Prior must be a dictionary with the keys {EXPECTED_PRIOR_KEYS}")

    nfs = Prior["nfs"]
    X_init = Prior["X_init"]
    F_init = Prior["F_init"]
    xk_in = Prior["xk_in"]

    # ----- STRICT ERROR CHECKING OF LOCAL VARIABLES
    # This raises exceptions on bad inputs and does not alter any arguments.
    checkinputss(Ffun, X_0, n, np_max, nf_max, g_tol, delta_0, nfs, m, X_init, F_init, xk_in, Low, Upp)

    # ------ FINALIZE OPTION LOCAL VARIABLES
    # This must be run after error checking main local variables, some of which set default values for options
    # NOTE: None of these local variables are submitted to error checking.
    if Options is None:
        Options = {}
    if not isinstance(Options, dict):
        raise TypeError("Error: Options argument must be a dictionary")
    if not set(Options).issubset(ALL_OPTIONS_KEYS):
        extras = set(Options).difference(ALL_OPTIONS_KEYS)
        raise ValueError(f"Error: Options dictionary contains unknown keys {extras}")
    if (("hfun" in Options) and ("combinemodels" not in Options)) or (("hfun" not in Options) and ("combinemodels" in Options)):
        raise ValueError("Error: Cannot provide only hfun or only combinemodels")

    defaults = compute_default_options(delta_0, g_tol, Low, Upp)
    for key, value in Options.items():
        defaults[key] = value

    printf = defaults["printf"]
    spsolver = defaults["spsolver"]
    delta_max = defaults["delta_max"]
    delta_min = defaults["delta_min"]
    delta_inact = defaults["delta_inact"]
    gamma_dec = defaults["gamma_dec"]
    gamma_inc = defaults["gamma_inc"]
    eta_1 = defaults["eta1"]
    hfun = defaults["hfun"]
    combinemodels = defaults["combinemodels"]

    solve_trsp = create_trsp_solver(spsolver)

    # ----- OPTIMIZE!
    eps = np.finfo(float).eps  # Define machine epsilon

    delta = delta_0
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
            print("%4i    Initial point  %11.5e" % (nf, hfun(F[nf, :])))
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
            for i in range(int(min(n - mp, nf_max - (nf + 1)))):
                nf += 1
                X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[i, :]))
                F[nf] = Ffun(X[nf])
                if np.any(np.isnan(F[nf])) or np.any(np.isinf(F[nf])):
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                    return X, F, hF, flag, xk_in
                hF[nf] = hfun(F[nf])
                if printf:
                    print("%4i   Geometry point  %11.5e" % (nf, hF[nf]))
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
            print("POUNDERS: nf: %4d; fval: %8e; delta: %8.3e; valid: %1i; mp: %2i; ng: %12.4e; ierror: %11.3e;" % (nf, hF[xk_in], delta, valid, mp, ng, ierror))
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
                        print("%4i   Critical point  %11.5e" % (nf, hF[nf]))
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
                    print("%4i   Model point     %11.5e" % (nf, hF[nf]))
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
