import numpy as np
from ibcdfo.pounders import _checkinputss as checkinputss

from .build_p_models import build_p_models
from .call_user_scripts import call_user_scripts
from .check_inputs_and_initialize import check_inputs_and_initialize
from .choose_generator_set import choose_generator_set
from .minimize_affine_envelope import minimize_affine_envelope
from .prepare_outputs_before_return import prepare_outputs_before_return

# # You'll need to uncomment the following two, and not have `eng = []` if you
# # want to use matlab's linprog in minimize_affine_envelope
# import matlab.engine
# eng = matlab.engine.start_matlab()
eng = []


def manifold_sampling_primal(hfun, Ffun, x0, Low, Upp, nf_max, subprob_switch):
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

    :param subprob_switch:
        Selects the trust-region subproblem solver/variant used internally.

    :return:
        Tuple ``(X, F, hF, xk_in, flag)`` where

        * ``X`` -- array of evaluated points in evaluation order
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

    # Deduce p from evaluating Ffun at x0
    try:
        F0 = Ffun(x0)
    finally:
        pass

    m = len(F0)
    nfs = 1
    X_init = np.atleast_2d(x0)
    F_init = np.atleast_2d(F0)

    n, delta, printf, fq_pars, tol, X, F, hF, Hash, nf, successful, xk_in, Hres, chi_k = check_inputs_and_initialize(x0, F0, nf_max)
    # This raises exceptions on bad inputs and does not alter any arguments.
    checkinputss(hfun, x0, n, fq_pars["npmax"], nf_max, tol["gtol"], delta, nfs, m, X_init, F_init, xk_in, Low, Upp, fq_pars["Par"])

    # Evaluate user scripts at x_0
    hF[nf], __, hashes_at_nf = hfun(F[nf])
    Hash[nf] = hashes_at_nf

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
                return prepare_outputs_before_return(X, F, hF, nf, xk_in, -1)
            if nf + 1 >= nf_max:
                return prepare_outputs_before_return(X, F, hF, nf, xk_in, 0)

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
                return prepare_outputs_before_return(X, F, hF, nf, xk_in, -2)

            # Line 8: Compute stationary measure chi_k
            Lows = np.maximum(Low - X[xk_in], -1.0)
            Upps = np.minimum(Upp - X[xk_in], 1.0)
            __, __, chi_k, __, lp_fail_flag = minimize_affine_envelope(hF[xk_in], f_bar, beta, G_k, np.zeros((n, n)), delta, Lows, Upps, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch, eng)
            if lp_fail_flag:
                return prepare_outputs_before_return(X, F, hF, nf, xk_in, -2)

            # Lines 9-11: Convergence test: tiny master model gradient and tiny delta
            if chi_k <= tol["gtol"] and delta <= tol["mindelta"]:
                return prepare_outputs_before_return(X, F, hF, nf, xk_in, chi_k)

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
        return prepare_outputs_before_return(X, F, hF, nf, xk_in, 0)
    else:
        return prepare_outputs_before_return(X, F, hF, nf, xk_in, chi_k)
