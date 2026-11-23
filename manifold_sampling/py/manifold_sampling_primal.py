import numpy as np
from ibcdfo.pounders import _checkinputss as checkinputss

from .build_p_models import build_p_models
from .call_user_scripts import call_user_scripts
from .check_inputs_and_initialize import check_inputs_and_initialize
from .choose_generator_set import choose_generator_set
from .defaults import ALL_OPTIONS_KEYS, EXPECTED_PRIOR_KEYS, compute_default_options
from .minimize_affine_envelope import minimize_affine_envelope
from .prepare_ms_outputs_before_return import prepare_ms_outputs_before_return

# # You'll need to uncomment the following two, and not have `eng = []` if you
# # want to use matlab's linprog in minimize_affine_envelope
# import matlab.engine
# eng = matlab.engine.start_matlab()
eng = []


def manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, Options=None, Prior=None):
    r"""
    Run manifold sampling to solve the composite nonsmooth optimization problem.

    :param hfun:
        Function implementing :math:`\hfun`. Supports two calling modes.

        **Mode 1**::

            hval, grads, hashes = hfun(z)

        where:

        * ``z`` is a length-:math:`\nd` array
        * ``hval`` is the scalar value :math:`\hfun(\zvec)`
        * ``grads`` is a :math:`(\nd, l)` array whose columns are gradients of the
          :math:`l` active selection functions at :math:`\zvec`
        * ``hashes`` is a list of identifiers for those :math:`l` active manifolds

        **Mode 2**::

            vals, grads = hfun(z, hashes)

        where the list ``hashes`` specifies which manifolds to evaluate, ``vals[i]`` is the
        value of the ``i`` th corresponding selection, and column ``i`` of
        ``grads`` is the gradient of the ``i`` th selection at :math:`\zvec`.

    :param Ffun: Function returning :math:`\Ffun(\psp)` as a length-:math:`\nd`
        array for a given length-:math:`\np` array ``x``.

    :param x0: Initial point (length :math:`\np` array).

    :param Low: Lower bounds for ``x`` (length :math:`\np` array).

    :param Upp: Upper bounds for ``x`` (length :math:`\np` array).

    :param nf_max: Maximum number of evaluations of ``Ffun``.

    :param Options: ``dict`` of method options.  Set to ``None`` to use default
        values.

        * **printf** - 0 for no printing to screen, 1 for standard output
          (default is 1)
        * **delta_0** - Initial trust-region radius (default is 0.1)
        * **delta_max** - Maximum allowed trust-region radius (default is
          :math:`10^8`)
        * **delta_min** - Minimum allowed trust-region radius; reaching this
          triggers termination (default is :math:`10^{-13}`)
        * **g_tol** - Tolerance on the stationary measure (default is
          :math:`10^{-13}`)
        * **eta_1** - In (0, 1); minimum ratio of actual to predicted
          reduction required to accept a step (default is 0.01)
        * **eta_3** - Ratio of actual to predicted reduction above which the
          trust-region radius may grow (default is 0.5)
        * **gamma_dec** - In (0, 1); factor by which the trust-region radius
          is decreased (default is 0.5)
        * **gamma_inc** - At least 1; factor by which the trust-region radius
          is increased (default is 2)
        * **hfun_test_mode** - Check the **hfun** implementation on every call
          (default is 1)
        * **subprob_switch** - Solver for the manifold sampling subproblem.
          Only ``"linprog"`` (default), which uses ``scipy.optimize.linprog``,
          is supported.

    :param Prior: ``dict`` describing past evaluations of ``Ffun``.  Set to
        ``None`` to run optimization assuming no past evaluations. A nonempty
        **Prior** must contain entries:

        * **nfs** - Number of past function evaluations
        * **X_init** - :math:`\mathrm{nfs} \times \np` NumPy array of distinct
          points :math:`\psp_k`
        * **F_init** - :math:`\mathrm{nfs} \times \nd` NumPy array of values
          :math:`\Ffun(\psp_k)` computed with ``Ffun``
        * **xk_in** - Zero-based index into ``X_init`` and ``F_init`` that
          corresponds to the point and value to use as the initial point for
          optimization. Note that if **Prior** is nonempty, **X_init[xk_in]**
          and **x0** must be identical.

        **nf_max** counts only new evaluations of ``Ffun``.

    :return:
        Tuple ``(X, F, hF, xk_in, flag)`` where

        * ``X`` -- array of the points in ``Prior`` followed by the evaluated
          points in evaluation order
        * ``F`` -- array with rows :math:`\Ffun(X[i])`
        * ``hF`` -- array with ``hF[i]`` = :math:`\hfun(\Ffun(X[i]))`
        * ``xk_in`` -- zero-based index in ``X`` of the final trust-region center
        * ``flag`` -- termination code (See general documentation)
    """
    # Some other values
    #  n:       [int]     Dimension of the domain of F (deduced from x0)
    #  p:       [int]     Dimension of the domain of h (deduced from evaluating F(x0))
    #  delta:   [dbl]     Positive starting trust region radius
    # Intermediate Variables:
    #   nf    [int]         Counter for the number of function evaluations
    #   s_k   [dbl]         Step from current iterate to approx. TRSP solution
    #   norm_g [dbl]        Stationary measure ||g||
    #   Gres [n x p]        Model gradients for each of the p outputs from Ffun
    #   Hres [n x n x p]    Model Hessians for each of the p outputs from Ffun
    #   Hash [cell]         Contains the hashes active at each evaluated point in X
    #   Act_Z_k [l cell]    List of hashes for active selection functions in TR
    #   G_k  [n x l]        Matrix of model gradients composed with gradients of elements in Act_Z_k
    #   D_k  [p x l_2]      Matrix of gradients of selection functions at different points in p-space

    # ----- EXTRACT ARGUMENTS & DEFINE DEFAULTS
    if Options is None:
        Options = {}
    if not isinstance(Options, dict):
        raise TypeError("Error: Options argument must be a dictionary")
    if not set(Options).issubset(ALL_OPTIONS_KEYS):
        extras = set(Options).difference(ALL_OPTIONS_KEYS)
        raise ValueError(f"Error: Options dictionary contains unknown keys {extras}")

    defaults = compute_default_options()
    for key, value in Options.items():
        defaults[key] = value

    subprob_switch = defaults["subprob_switch"]
    if subprob_switch != "linprog":
        raise ValueError(f"Error: Options subprob_switch must be 'linprog' ({subprob_switch})")

    # -- Prior dictionary
    if Prior is None:
        # Deduce p from evaluating Ffun at x0
        try:
            F0 = Ffun(x0)
        finally:
            pass

        nfs = 1
        X_init = np.atleast_2d(x0)
        F_init = np.atleast_2d(F0)
        xk_in = 0
    else:
        if not isinstance(Prior, dict):
            raise TypeError("Error: Prior argument must be a dictionary")
        if set(Prior) != EXPECTED_PRIOR_KEYS:
            raise ValueError(f"Error: Prior must be a dictionary with the keys {EXPECTED_PRIOR_KEYS}")

        nfs = Prior["nfs"]
        X_init = Prior["X_init"]
        F_init = Prior["F_init"]
        xk_in = Prior["xk_in"]
        nf_max = nf_max + nfs

    m = F_init.shape[1]

    n, delta, printf, fq_pars, tol, successful, Hres, chi_k = check_inputs_and_initialize(x0, m, nf_max, defaults)
    # This raises exceptions on bad inputs and does not alter any arguments.
    checkinputss(hfun, x0, n, fq_pars["npmax"], nf_max, tol["gtol"], delta, nfs, m, X_init, F_init, xk_in, Low, Upp, fq_pars["Par"])

    X = np.vstack((X_init, np.zeros((nf_max - nfs, n))))
    F = np.vstack((F_init, np.zeros((nf_max - nfs, m))))
    hF = np.zeros((nf_max, 1))
    Hash = {}
    nf = nfs - 1

    # Evaluate hfun at the points already evaluated
    for i in range(nf + 1):
        hF[i], __, Hash[i] = hfun(F[i])

    H_mm = np.zeros((n, n))

    while nf + 1 < nf_max and delta > tol["mindelta"]:
        if printf:
            print("MSP: nf: %4d; fval: %8e; delta: %8.3e; chi: %8.3e;" % (nf, np.squeeze(hF[xk_in]), delta, chi_k))

        bar_delta = delta

        successful = False
        # Line 3: manifold sampling while loop
        while nf + 1 < nf_max:
            # Line 4: build models
            Gres, Hres, X, F, hF, nf, Hash = build_p_models(nf, nf_max, xk_in, delta, F, X, hF, Hres, fq_pars, tol, hfun, Ffun, Hash, Low, Upp)
            if len(Gres) == 0:
                return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, -1, printf, chi_k)
            if nf + 1 >= nf_max:
                return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, 1, printf, chi_k)  # Budget exceeded

            # Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
            D_k, Act_Z_k, f_bar = choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun)
            G_k = Gres @ D_k
            beta = np.maximum(0, f_bar - hF[xk_in])

            # Line 6: Choose Hessians
            H_k = np.zeros((G_k.shape[1], n + 1, n + 1))
            for i in range(G_k.shape[1]):
                for j in range(Hres.shape[2]):
                    H_k[i, 1:, 1:] = np.squeeze(H_k[i, 1:, 1:]) + D_k[j, i] * Hres[:, :, j]

            # Line 7: Find a candidate s_k by solving QP
            Lows = np.maximum(Low - X[xk_in], -delta)
            Upps = np.minimum(Upp - X[xk_in], delta)
            s_k, tau_k, __, lambda_k, lp_fail_flag = minimize_affine_envelope(hF[xk_in], f_bar, beta, G_k, H_mm, delta, Lows, Upps, H_k, subprob_switch, eng)
            if lp_fail_flag:
                return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, -2, printf, chi_k)

            # Line 8: Compute stationary measure chi_k
            Lows = np.maximum(Low - X[xk_in], -1.0)
            Upps = np.minimum(Upp - X[xk_in], 1.0)
            __, __, chi_k, __, lp_fail_flag = minimize_affine_envelope(hF[xk_in], f_bar, beta, G_k, np.zeros((n, n)), delta, Lows, Upps, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch, eng)
            if lp_fail_flag:
                return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, -2, printf, chi_k)

            # Lines 9-11: Convergence test: tiny master model gradient and tiny delta
            if chi_k <= tol["gtol"] and delta <= tol["mindelta"]:
                return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, 0, printf, chi_k)

            # Line 12: Evaluate F
            nf, X, F, hF, Hash, hashes_at_nf = call_user_scripts(nf, X, F, hF, Hash, Ffun, hfun, X[xk_in] + np.transpose(s_k), tol, Low, Upp, 1)
            # Line 13: Compute rho_k
            ared = hF[xk_in] - hF[nf]
            pred = -tau_k
            if pred == 0:
                rho_k = -np.inf
            else:
                rho_k = ared / pred

            # Lines 14-16: Check for success
            if rho_k >= tol["eta1"] and pred > 0:
                successful = True
                break
            else:
                # Line 18: Check temporary activities after adding TRSP solution to X
                __, tmp_Act_Z_k, __ = choose_generator_set(X, Hash, xk_in, nf, delta, F, hfun)

                # Lines 19: See if any new activities
                if np.all(np.isin(tmp_Act_Z_k, Act_Z_k)):
                    # Line 20: See if intersection is nonempty
                    if np.any(np.isin(hashes_at_nf, Act_Z_k)):
                        break
                    else:
                        # Line 24: Shrink delta
                        delta = tol["gamma_dec"] * delta

        if successful:
            xk_in = nf
            if rho_k > tol["eta3"] and np.linalg.norm(s_k, ord=np.inf) > 0.8 * bar_delta:
                # Update delta if rho is sufficiently large
                delta = bar_delta * tol["gamma_inc"]
                # h_activity_tol = min(1e-8, delta);
        else:
            # Line 21: iteration is unsuccessful; shrink Delta
            delta = max(bar_delta * tol["gamma_dec"], tol["mindelta"])
            # h_activity_tol = min(1e-8, delta);

    if nf + 1 >= nf_max:
        return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, 1, printf, chi_k)  # Budget exceeded
    elif chi_k <= tol["gtol"]:
        return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, 0, printf, chi_k)
    else:
        return prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, -6, printf, chi_k)  # delta_min reached without satisfying g_tol
