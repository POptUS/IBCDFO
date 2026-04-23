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


def manifold_sampling_primal(hfun, Ffun, x0, L, U, nf_max, subprob_switch):
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

    :param L: Lower bounds for ``x`` (length :math:`\np` array).

    :param U: Upper bounds for ``x`` (length :math:`\np` array).

    :param nf_max: Maximum number of evaluations of ``Ffun``.

    :param subprob_switch:
        Selects the trust-region subproblem solver/variant used internally.

    :return:
        Tuple ``(X, F, h, xkin, flag)`` where

        * ``X`` -- array of evaluated points in evaluation order
        * ``F`` -- array with rows :math:`\Ffun(X[i])`
        * ``h`` -- array with :math:`h[i] = \hfun(\Ffun(X[i]))`
        * ``xkin`` -- zero-based index in ``X`` of the final trust-region center
        * ``flag`` -- termination code

    See User Guide for interpretation of flag values.
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

    n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres = check_inputs_and_initialize(x0, F0, nf_max)
    flag, x0, __, F0, L, U, xkin = checkinputss(hfun, np.atleast_2d(x0), n, fq_pars["npmax"], nf_max, tol["gtol"], delta, 1, len(F0), np.atleast_2d(x0), np.atleast_2d(F0), xkin, L, U)
    if flag == -1:
        print("MSP: Error with inputs. Exiting.")
        X = x0
        F = F0
        h = []
        return X, F, h, xkin, flag

    # Evaluate user scripts at x_0
    h[nf], __, hashes_at_nf = hfun(F[nf])
    Hash[nf] = hashes_at_nf

    H_mm = np.zeros((n, n))

    while nf + 1 < nf_max and delta > tol["mindelta"]:
        bar_delta = delta

        # Line 3: manifold sampling while loop
        while nf + 1 < nf_max:
            # Line 4: build models
            Gres, Hres, X, F, h, nf, Hash = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
            if len(Gres) == 0:
                return prepare_outputs_before_return(X, F, h, nf, xkin, -1)
            if nf + 1 >= nf_max:
                return prepare_outputs_before_return(X, F, h, nf, xkin, 0)

            # Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
            D_k, Act_Z_k, f_bar = choose_generator_set(X, Hash, xkin, nf, delta, F, hfun)
            G_k = Gres @ D_k
            beta = np.maximum(0, f_bar - h[xkin])

            # Line 6: Choose Hessians
            H_k = np.zeros((G_k.shape[1], n + 1, n + 1))
            for i in range(G_k.shape[1]):
                for j in range(Hres.shape[2]):
                    H_k[i, 1:, 1:] = np.squeeze(H_k[i, 1:, 1:]) + D_k[j, i] * Hres[:, :, j]

            # Line 7: Find a candidate s_k by solving QP
            Low = np.maximum(L - X[xkin], -delta)
            Upp = np.minimum(U - X[xkin], delta)
            s_k, tau_k, __, lambda_k, lp_fail_flag = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch, eng)
            if lp_fail_flag:
                return prepare_outputs_before_return(X, F, h, nf, xkin, -2)

            # Line 8: Compute stationary measure chi_k
            Low = np.maximum(L - X[xkin], -1.0)
            Upp = np.minimum(U - X[xkin], 1.0)
            __, __, chi_k, __, lp_fail_flag = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, np.zeros((n, n)), delta, Low, Upp, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch, eng)
            if lp_fail_flag:
                return prepare_outputs_before_return(X, F, h, nf, xkin, -2)

            # Lines 9-11: Convergence test: tiny master model gradient and tiny delta
            if chi_k <= tol["gtol"] and delta <= tol["mindelta"]:
                return prepare_outputs_before_return(X, F, h, nf, xkin, chi_k)

            # Line 12: Evaluate F
            nf, X, F, h, Hash, hashes_at_nf = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin] + np.transpose(s_k), tol, L, U, 1)
            # Line 13: Compute rho_k
            ared = h[xkin] - h[nf]
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
                __, tmp_Act_Z_k, __ = choose_generator_set(X, Hash, xkin, nf, delta, F, hfun)

                # Lines 19: See if any new activities
                if np.all(np.isin(tmp_Act_Z_k, Act_Z_k)):
                    # Line 20: See if intersection is nonempty
                    if np.any(np.isin(hashes_at_nf, Act_Z_k)):
                        successful = False
                        break
                    else:
                        # Line 24: Shrink delta
                        delta = tol["gamma_dec"] * delta

        if successful:
            xkin = nf
            if rho_k > tol["eta3"] and np.linalg.norm(s_k, ord=np.inf) > 0.8 * bar_delta:
                # Update delta if rho is sufficiently large
                delta = bar_delta * tol["gamma_inc"]
                # h_activity_tol = min(1e-8, delta);
        else:
            # Line 21: iteration is unsuccessful; shrink Delta
            delta = max(bar_delta * tol["gamma_dec"], tol["mindelta"])
            # h_activity_tol = min(1e-8, delta);
        if printf:
            print("MSP: nf: %8d; fval: %8e; chi: %8e; radius: %8e;" % (nf, np.squeeze(h[xkin]), chi_k, delta))

    if nf + 1 >= nf_max:
        return prepare_outputs_before_return(X, F, h, nf, xkin, 0)
    else:
        return prepare_outputs_before_return(X, F, h, nf, xkin, chi_k)
