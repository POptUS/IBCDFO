# This code solves the problem
#     minimize h(F(x))
# where x is an [n by 1] vector, F is a blackbox function mapping from R^n to
# R^p, and h is a nonsmooth function mapping from R^p to R.
#
#
# Inputs:
#  hfun:    [func handle] Evaluates h, returning the [scalar] function
#                         value and [k x m] subgradients for all k limiting
#                         gradients at the point given.
#  Ffun:    [func handle] Evaluates F, the black box simulation, returning
#                         a [1 x m] vector.
#  nfmax:   [int]         Maximum number of function evaluations.
#  x0:      [1 x n dbl]   Starting point.
#  L:       [1 x n dbl]   Lower bounds.
#  U:       [1 x n dbl]   Upper bounds.
#  GAMS_options:
#  subprob_switch:
#
# Outputs:
#   X:      [nfmax x n]   Points evaluated
#   F:      [nfmax x p]   Their simulation values
#   h:      [nfmax x 1]   The values h(F(x))
#   xkin:   [int]         Current trust region center

import numpy as np
import sys
from check_inputs_and_initialize import check_inputs_and_initialize
from call_user_scripts import call_user_scripts
from checkinputss import checkinputss
from build_p_models import build_p_models
from save_quadratics_call_GAMS import save_quadratics_call_GAMS
from choose_generator_set import choose_generator_set

def goombah(hfun, Ffun, nfmax, x0, L, U, GAMS_options, subprob_switch):
    try:
        F0 = Ffun(x0)
        F0 = F0.flatten()
    except:
        print('Problem using Ffun. Exiting')
        X, F, h, xkin = [], [], [], []
        flag = -1
        return X, F, h, xkin, flag

    n, delta, printf, fq_pars, tol, X, F, h, Hash, nf, successful, xkin, Hres = check_inputs_and_initialize(x0, F0, nfmax)
    flag, x0, __, F0, L, U, xkin = checkinputss(hfun, x0, n, fq_pars["npmax"], nfmax, tol["gtol"], delta, 1, len(F0), F0, xkin, L, U)
    
    h[nf], _, hashes_at_nf = hfun(F[nf, :])
    Hash[nf] = hashes_at_nf
    
    I_n = np.eye(n)
    for i in range(n):
        nf, X, F, h, Hash, _ = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + delta * I_n[i, :], tol, L, U, 1)
    
    H_mm = np.zeros((n, n))
    beta_exp = 1.0
    
    while nf < nfmax and delta > tol['mindelta']:
        xkin = int(np.argmin(h[:nf]))
        
        Gres, Hres, X, F, h, nf, Hash = build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
        
        if len(Gres) == 0:
            print('Empty Gres. Delta =', delta)
            X = X[:nf, :]
            F = F[:nf, :]
            h = h[:nf, :]
            return X, F, h, xkin
        
        if nf >= nfmax:
            return X, F, h, xkin
        
        Low = np.maximum(L - X[xkin, :], -delta)
        Upp = np.minimum(U - X[xkin, :], delta)
        
        sk, pred_dec = save_quadratics_call_GAMS(Hres, Gres, F[xkin, :], Low, Upp, X[xkin, :], X[xkin, :], h[xkin], GAMS_options, hfun)
        
        if pred_dec == 0:
            rho_k = -np.inf
        else:
            nf, X, F, h, Hash, _ = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + sk, tol, L, U, 1)
            rho_k = (h[xkin] - h[nf]) / min(1.0, delta**(1.0 + beta_exp))
        
        if rho_k > tol["eta1"]:
            if np.linalg.norm(X[xkin, :] - X[nf, :], np.inf) >= 0.8 * delta:
                delta = delta * tol["gamma_inc"]
            xkin = nf
        else:
            # Need to do one MS-P loop
            
            bar_delta = delta
            
            while nf < nfmax:
                Gres, Hres, X, F, h, nf, Hash = build_p_models(nf, nfmax, xkin, delta, F, X, h, Hres, fq_pars, tol, hfun, Ffun, Hash, L, U)
                
                if len(Gres) == 0:
                    print('Model building failed. Empty Gres. Delta =', delta)
                    X = X[:nf, :]
                    F = F[:nf, :]
                    h = h[:nf, :]
                    flag = -1
                    return X, F, h, flag
                
                if nf >= nfmax:
                    return X, F, h, xkin
                
                import ipdb; ipdb.set_trace()
                D_k, Act_Z_k, f_bar = choose_generator_set(X, Hash, 3, xkin, nf, delta, F, hfun)
                G_k = np.dot(Gres, D_k)
                beta = max(0, f_bar - h[xkin])
                
                H_k = np.zeros((G_k.shape[1], n + 1, n + 1))
                for i in range(G_k.shape[1]):
                    for j in range(Hres.shape[2]):
                        H_k[i, 1:, 1:] += D_k[j, i] * Hres[:, :, j]
                
                Low = np.maximum(L - X[xkin, :], -delta)
                Upp = np.minimum(U - X[xkin, :], delta)
                
                s_k, tau_k = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, H_mm, delta, Low, Upp, H_k, subprob_switch)
                
                Low = np.maximum(L - X[xkin, :], -1.0)
                Upp = np.minimum(U - X[xkin, :], 1.0)
                _, _, chi_k = minimize_affine_envelope(h[xkin], f_bar, beta, G_k, np.zeros((n, n)), delta, Low, Upp, np.zeros((G_k.shape[1], n + 1, n + 1)), subprob_switch)
                
                if chi_k <= tol["gtol"] and delta <= tol["mindelta"]:
                    print('Convergence satisfied: small stationary measure and small delta')
                    X = X[:nf, :]
                    F = F[:nf, :]
                    h = h[:nf, :]
                    flag = chi_k
                    return X, F, h, flag
                
                if printf:
                    trsp_fun = lambda x: max_affine(x, h[xkin], f_bar, beta, G_k, H_mm)
                    # plot_again_j(X, xkin, delta, s_k, [], nf, trsp_fun, L, U)
                
                nf, X, F, h, Hash, hashes_at_nf = call_user_scripts(nf, X, F, h, Hash, Ffun, hfun, X[xkin, :] + s_k.T, tol, L, U, 1)
                
                ared = h[xkin] - h[nf]
                pred = -tau_k
                rho_k = ared / pred
                
                if rho_k >= tol["eta1"] and pred > 0:
                    successful = True
                    break
                else:
                    tmp_Act_Z_k = choose_generator_set(X, Hash, 3, xkin, nf, delta, F, hfun)[1]
                    
                    if all(item in tmp_Act_Z_k for item in Act_Z_k):
                        if any(item in hashes_at_nf for item in Act_Z_k):
                            successful = False
                            break
                        else:
                            delta = tol["gamma_dec"] * delta
            if successful:
                xkin = nf
                if rho_k > tol["eta3"] and np.linalg.norm(s_k, np.inf) > 0.8 * bar_delta:
                    delta = bar_delta * tol["gamma_inc"]
            else:
                delta = max(bar_delta * tol["gamma_dec"], tol["mindelta"])
        
        print(f'nf: {nf:8d}; fval: {h[xkin,0]:8e}; radius: {delta:8e};')
    
    if nf >= nfmax:
        flag = 0
    else:
        X = X[:nf, :]
        F = F[:nf, :]
        h = h[:nf, :]
    
    return X, F, h, xkin, flag
