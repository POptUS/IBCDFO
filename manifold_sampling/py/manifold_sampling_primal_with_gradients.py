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
#  Ffun:    [func]    Evaluates F, the black box simulation, returning a [1 x p] vector and
#                     a [p x n] Jacobian
#  x0:      [1 x n]   Starting point
#  nfmax:   [int]     Maximum number of function evaluations
#
# Outputs:
#   X:      [nfmax x n]   Points evaluated
#   F:      [nfmax x p]   Their simulation values
#   Grad:   [nfmax x p x n] Their gradient values
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
import ipdb

from .build_p_models import build_p_models
from .call_user_scripts_with_gradients import call_user_scripts
from .check_inputs_and_initialize_with_gradients import check_inputs_and_initialize
from .choose_generator_set import choose_generator_set
from .minimize_affine_envelope import minimize_affine_envelope
from .prepare_outputs_before_return_with_gradients import prepare_outputs_before_return
from .overapproximating_hessian import overapproximating_hessian

# # You'll need to uncomment the following two, and not have `eng = []` if you
# # want to use matlab's linprog in minimize_affine_envelope
# import matlab.engine
# eng = matlab.engine.start_matlab()
eng = []


def manifold_sampling_primal_with_gradients(hfun, Ffun, x0, L, U, nfmax, subprob_switch, gtol=None, printf=False):
    # Deduce p from evaluating Ffun at x0
    try:
        F0, Grad0 = Ffun(x0)
    finally:
        pass

    n, delta, _, fq_pars, tol, X, F, Grad, h, Hash, nf, successful, xkin, Hres = check_inputs_and_initialize(x0, F0, Grad0, nfmax)

    if gtol is not None:
        tol["gtol"] = gtol
        tol["mindelta"] = gtol
        delta = np.maximum(delta, 2 * tol["mindelta"])

    flag, x0, __, F0, L, U, xkin = checkinputss(hfun, np.atleast_2d(x0), n, fq_pars["npmax"], nfmax, tol["gtol"], delta, 1, len(F0), np.atleast_2d(x0), np.atleast_2d(F0), xkin, L, U)
    if flag == -1:
        if printf:
            print("MSP: Error with inputs. Exiting.")
        X = x0
        F = F0
        Grad = Grad0
        h = []
        return X, F, Grad, h, xkin, flag

    # Evaluate user scripts at x_0
    h[nf], __, hashes_at_nf = hfun(F[nf])
    Hash[nf] = hashes_at_nf

    H_mm = 1e-8 * np.eye(n)

    while nf + 1 < nfmax and delta > tol["mindelta"]:
        bar_delta = delta

        # Line 3: manifold sampling while loop
        while nf + 1 < nfmax:
            # Line 4: build models
            Gres = Grad[xkin].T
            if len(Gres) == 0:
                return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, -1)
            if nf + 1 >= nfmax:
                return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, 0)

            # Line 5: Build set of activities Act_Z_k, gradients D_k, G_k, and beta
            D_k, Act_Z_k, f_bar = choose_generator_set(X, Hash, tol["gentype"], xkin, nf, delta, F, hfun)
            G_k = Gres @ D_k
            beta = np.maximum(0, f_bar - h[xkin])

            # Line 6: Choose Hessians (in this code: a nontrivial master model Hessian!)
            # this does nothing with subprob_switch==quadprog, but I don't want to change the minimize_affine_envelope API right now.
            H_k = np.zeros((G_k.shape[1], n + 1, n + 1))

            H_mm = 1e-8 * np.eye(n)
            H_mm = overapproximating_hessian(H_mm, X, nf, xkin, h, G_k, f_bar, beta, delta)

            # Line 7: Find a candidate s_k by solving QP
            Low = np.maximum(L - X[xkin], -delta)
            Upp = np.minimum(U - X[xkin], delta)
            s_k, tau_k, __, lambda_k, lp_fail_flag = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch, eng)
            if lp_fail_flag:
                return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, -2)

            # Line 8: Compute stationary measure chi_k
            Low = np.maximum(L - X[xkin], -1.0)
            Upp = np.minimum(U - X[xkin], 1.0)
            __, __, chi_k, __, lp_fail_flag = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, np.zeros((n, n)), delta, Low, Upp, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch, eng)
            if lp_fail_flag:
                return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, -2)

            # Lines 9-11: Convergence test: tiny master model gradient and tiny delta
            if chi_k <= tol["gtol"] and delta <= tol["mindelta"]:
                return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, chi_k)

            # Line 12: Evaluate F
            nf, X, F, Grad, h, Hash, hashes_at_nf = call_user_scripts(nf, X, F, Grad, h, Hash, Ffun, hfun, np.squeeze(X[xkin] + s_k.T), tol, L, U, 1)
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
            #print("max eig of H_mm: ", np.amax(np.linalg.eig(H_mm)[0]))

    if nf + 1 >= nfmax:
        return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, 0)
    else:
        return prepare_outputs_before_return(X, F, Grad, h, nf, xkin, chi_k)
