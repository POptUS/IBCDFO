"""
This tests pounder (no structure) against pounders with a novel hfun from
quantum. The objective is a function of a real vector gamma and a complex
vector output G_of_gamma. Given these, the objective is
imag(sum_{i=1}^m gamma_i*G_of_gamma**2), which is
sum_{i=1}^m imag(gamma_i*G_of_gamma**2), which is
sum_{i=1}^m gamma_i*imag(G_of_gamma**2), which is
sum_{i=1}^m 2 * gamma_i*G_of_gamma_r*G_of_gamma_i

This is the hfun. So given gamma, the Ffun computes G_of_gamma and returns its
real and imaginary parts.
"""

import ibcdfo.pounders as pdrs
import numpy as np
from declare_hfun_and_combine_model_with_jax import hfun, combinemodels_jax  

nf_max = 500
g_tol = 1e-5
n = 5
X_0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Low = -2 * np.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
Upp = 2 * np.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
nfs = 1
delta = 0.1

def Ffun(gamma, nostruct=True):
    G_of_gamma = np.sin(gamma) - np.arange(1, len(gamma) + 1) * np.cos(gamma) * 1j
    out = np.squeeze(np.hstack((gamma, np.real(G_of_gamma), np.imag(G_of_gamma))))
    if nostruct:
        return hfun(out)
    else:
        return out


hF = {}
for call in range(2):
    if call == 0:
        # This calls pounders with m=1 (not using structure)
        Ffun_to_use = lambda gamma: Ffun(gamma,True)
        m = 1  # not using structure
        Opts = {
                "hfun": lambda F: np.squeeze(F),  # not using structure
                "combinemodels": pdrs.identity_combine,  # not using structure
                }
    elif call == 1:
        # This uss jax to get models of the hFun and we call pounders using structure
        Ffun_to_use = lambda gamma: Ffun(gamma,False)
        m = 3 * n  # using structure
        Opts = {
        "hfun": hfun,  # using structure
        "combinemodels": combinemodels_jax,  # using structure
        }

    [_, _, hF[call], flag, _] = pdrs.pounders(Ffun_to_use, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
    assert flag == 0, "Didn't reach critical point"


print(f"Using structure uses {len(hF[1])} evals. Not using structure uses {len(hF[0])}")
assert len(hF[1]) < len(hF[0]), "While not for every problem, using structure on this problem should be beneficial"
