# This code solves the problem
#     minimize h(F(x))
# where x is an [n by 1] vector, F is a blackbox function mapping from R^n to
# R^p, and h is a nonsmooth function mapping from R^p to R.
#
#
# Inputs:
#  hfun:    [func]   Given point z, returns
#                      - [scalar] the value h(z)
#                      - [p x l] gradients for all l limiting gradients at z
#                      - [1 x l list of hashes] for each manifold active at z
#                    Given point z and l hashes H, returns
#                      - [1 x l] the value h_i(z) for each hash in H
#                      - [p x l] gradients of h_i(z) for each hash in H
#  Ffun:    [func]    Evaluates F, the black box simulation, returning a [1 x p] vector.
#  x0:      [1 x n]   Starting point
#  nfmax:   [int]     Maximum number of function evaluations
#
# Outputs:
#   X:      [nfmax x n]   Points evaluated
#   F:      [nfmax x p]   Their simulation values
#   h:      [nfmax x 1]   The values h(F(x))
#   xkin:   [int]         Current trust region center
#   flag:   [int]         Inform user why we stopped.
#                           -1 if error
#                            0 if nfmax function evaluations were performed
#                            final model gradient norm otherwise
#
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

import numpy as np
from ibcdfo.pounders import checkinputss

from .build_p_models import build_p_models
from .call_user_scripts import call_user_scripts
from .check_inputs_and_initialize import check_inputs_and_initialize
from .choose_generator_set import choose_generator_set
from .minimize_affine_envelope import minimize_affine_envelope
from .prepare_outputs_before_return import prepare_outputs_before_return


def manifold_sampling_primal(hfun, Ffun, x0, L, U, nfmax, subprob_switch):
    # Deduce p from evaluating Ffun at x0
    try:
        F0 = Ffun(x0)
    finally:
        pass

    n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres = check_inputs_and_initialize(x0, F0, nfmax)
    flag, x0, __, F0, L, U, xkin = checkinputss(hfun, np.atleast_2d(x0), n, fq_pars["npmax"], nfmax, tol["gtol"], delta, 1, len(F0), np.atleast_2d(x0), np.atleast_2d(F0), xkin, L, U)
    if flag == -1:
        if printf:
            print("MSP: Error with inputs. Exiting.")
        X = x0
        F = F0
        h = []
        return X, F, h, xkin, flag

    # Evaluate user scripts at x_0
    h[nf], __, hashes_at_nf = hfun(F[nf])
    Hash[nf] = hashes_at_nf

    H_mm = np.zeros((n, n))

    while nf + 1 < nfmax and delta > tol["mindelta"]:
        bar_delta = delta

        # Line 3: manifold sampling while loop
        while nf + 1 < nfmax:
            # Line 4: build models
            Gres, Hres, X, F, h, nf, Hash = build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
            if len(Gres) == 0:
                return prepare_outputs_before_return(X, F, h, nf, xkin, -1)
            if nf + 1 >= nfmax:
                return prepare_outputs_before_return(X, F, h, nf, xkin, 0)

            # Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
            D_k, Act_Z_k, f_bar = choose_generator_set(X, Hash, tol["gentype"], xkin, nf, delta, F, hfun)
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
            s_k, tau_k, __, lambda_k = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch)

            # Line 8: Compute stationary measure chi_k
            Low = np.maximum(L - X[xkin], -1.0)
            Upp = np.minimum(U - X[xkin], 1.0)
            __, __, chi_k, __ = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, np.zeros((n, n)), delta, Low, Upp, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch)

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
                __, tmp_Act_Z_k, __ = choose_generator_set(X, Hash, tol["gentype"], xkin, nf, delta, F, hfun)

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
            print("MSP: nf: %8d; fval: %8e; chi: %8e; radius: %8e;" % (nf, h[xkin], chi_k, delta))

    if nf + 1 >= nfmax:
        return prepare_outputs_before_return(X, F, h, nf, xkin, 0)
    else:
        return prepare_outputs_before_return(X, F, h, nf, xkin, chi_k)
