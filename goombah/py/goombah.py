import warnings

import numpy as np
from ibcdfo.manifold_sampling.build_p_models import build_p_models
from ibcdfo.manifold_sampling.call_user_scripts import call_user_scripts
from ibcdfo.manifold_sampling.check_inputs_and_initialize import check_inputs_and_initialize
from ibcdfo.manifold_sampling.choose_generator_set import choose_generator_set
from ibcdfo.manifold_sampling.minimize_affine_envelope import minimize_affine_envelope
from ibcdfo.manifold_sampling.prepare_ms_outputs_before_return import prepare_ms_outputs_before_return
from ibcdfo.pounders import _checkinputss as checkinputss

# # You'll need to uncomment the following two, and not have `eng = []` if you
# # want to use matlab's linprog in minimize_affine_envelope
# import matlab.engine
# eng = matlab.engine.start_matlab()
eng = []


def goombah(hfun, Ffun, nf_max, x0, L, U, trsp_solver, subprob_switch="linprog"):
    r"""
    Solve :math:`\min_{\xvec} \hfun(\Ffun(\xvec))` using GOOMBAH: a
    trust-region method that, at each iterate, first tries a single
    trust-region subproblem (TRSP) step built from quadratic models of all
    :math:`p` components of :math:`\Ffun`, falling back to a manifold
    sampling (MS-P) iteration when that step is unsuccessful.

    :param hfun: Function evaluating :math:`\hfun`, as described in
        :py:func:`ibcdfo.manifold_sampling.manifold_sampling_primal`.
    :param Ffun: Function returning :math:`\Ffun(\xvec)` as a length-:math:`p`
        array for a given length-:math:`n` array ``x``.
    :param nf_max: Maximum number of evaluations of ``Ffun``.
    :param x0: Initial point (length-:math:`n` array).
    :param L: Lower bounds for ``x`` (length-:math:`n` array).
    :param U: Upper bounds for ``x`` (length-:math:`n` array).
    :param trsp_solver: TRSP solver with the interface created by
        :py:func:`create_goombah_trsp_solver`.
    :param subprob_switch: Selects the TRSP solver/variant used internally by
        the manifold sampling fallback.

    :return: Tuple ``(X, F, h, xkin, flag)``, as described in
        :py:func:`ibcdfo.manifold_sampling.manifold_sampling_primal`.
    """
    try:
        F0 = Ffun(x0)
    except Exception:
        warnings.warn("Problem using Ffun. Exiting")
        return np.array([]), np.array([]), np.array([]), None, -1

    m = len(F0)
    nfs = 1
    X_init = np.atleast_2d(x0)
    F_init = np.atleast_2d(F0)

    n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres, chi_k = check_inputs_and_initialize(x0, F0, nf_max)
    # This raises exceptions on bad inputs and does not alter any arguments.
    checkinputss(hfun, x0, n, fq_pars["npmax"], nf_max, tol["gtol"], delta, nfs, m, X_init, F_init, xkin, L, U, fq_pars["Par"])

    h[nf], __, hashes_at_nf = hfun(F[nf])
    Hash[nf] = hashes_at_nf

    I_n = np.eye(n)
    for i in range(n):
        nf, X, F, h, Hash, __ = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin] + delta * I_n[i], tol, L, U, 1)

    H_mm = np.zeros((n, n))
    beta_exp = 1.0

    while nf + 1 < nf_max and delta > tol["mindelta"]:
        if printf:
            print("nf: %8d; fval: %8e; radius: %8e;" % (nf, np.squeeze(h[xkin]), delta))

        xkin = int(np.argmin(h[: nf + 1]))

        # Line: build p component models
        Gres, Hres, X, F, h, nf, Hash = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
        if len(Gres) == 0:
            return prepare_ms_outputs_before_return(X, F, h, nf, xkin, -1, printf, chi_k)
        if nf + 1 >= nf_max:
            return prepare_ms_outputs_before_return(X, F, h, nf, xkin, 1, printf, chi_k)  # Budget exceeded

        # Try a single GOOMBAH TRSP step built directly from the p models
        Low = np.maximum(L - X[xkin], -delta)
        Upp = np.minimum(U - X[xkin], delta)
        s_k, pred_dec = trsp_solver(Hres, Gres, F[xkin], Low, Upp, h[xkin], hfun)
        if pred_dec == 0:
            rho_k = -np.inf
        else:
            nf, X, F, h, Hash, __ = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin] + s_k, tol, L, U, 1)
            rho_k = (h[xkin] - h[nf]) / min(1.0, delta ** (1.0 + beta_exp))

        if rho_k > tol["eta1"]:  # Success with the GOOMBAH step
            if np.linalg.norm(X[xkin] - X[nf], ord=np.inf) >= 0.8 * delta:
                delta = delta * tol["gamma_inc"]
            xkin = nf
        else:
            # The GOOMBAH step was unsuccessful: do one MS-P loop
            bar_delta = delta
            successful = False

            while nf + 1 < nf_max:
                # Line 4: build models
                Gres, Hres, X, F, h, nf, Hash = build_p_models(nf, nf_max, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
                if len(Gres) == 0:
                    return prepare_ms_outputs_before_return(X, F, h, nf, xkin, -1, printf, chi_k)
                if nf + 1 >= nf_max:
                    return prepare_ms_outputs_before_return(X, F, h, nf, xkin, 1, printf, chi_k)  # Budget exceeded

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
                    return prepare_ms_outputs_before_return(X, F, h, nf, xkin, -2, printf, chi_k)

                # Line 8: Compute stationary measure chi_k
                Low = np.maximum(L - X[xkin], -1.0)
                Upp = np.minimum(U - X[xkin], 1.0)
                __, __, chi_k, __, lp_fail_flag = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, np.zeros((n, n)), delta, Low, Upp, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch, eng)
                if lp_fail_flag:
                    return prepare_ms_outputs_before_return(X, F, h, nf, xkin, -2, printf, chi_k)

                # Lines 9-11: Convergence test: tiny master model gradient and tiny delta
                if chi_k <= tol["gtol"] and delta <= tol["mindelta"]:
                    return prepare_ms_outputs_before_return(X, F, h, nf, xkin, 0, printf, chi_k)

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

                    # Line 19: See if any new activities
                    if np.all(np.isin(tmp_Act_Z_k, Act_Z_k)):
                        # Line 20: See if intersection is nonempty
                        if np.any(np.isin(hashes_at_nf, Act_Z_k)):
                            break
                        else:
                            # Line 24: Shrink delta
                            delta = tol["gamma_dec"] * delta

            if successful:
                xkin = nf  # Line 15: Update TR center and radius
                if rho_k > tol["eta3"] and np.linalg.norm(s_k, ord=np.inf) > 0.8 * bar_delta:
                    delta = bar_delta * tol["gamma_inc"]
            else:
                # Line 21: iteration is unsuccessful; shrink Delta
                delta = max(bar_delta * tol["gamma_dec"], tol["mindelta"])

    if nf + 1 >= nf_max:
        return prepare_ms_outputs_before_return(X, F, h, nf, xkin, 1, printf, chi_k)  # Budget exceeded
    elif chi_k <= tol["gtol"]:
        return prepare_ms_outputs_before_return(X, F, h, nf, xkin, 0, printf, chi_k)
    else:
        return prepare_ms_outputs_before_return(X, F, h, nf, xkin, -6, printf, chi_k)  # delta_min reached without satisfying g_tol
