"""
This tests pounder (no structure) against pounders with a novel hfun arising in a 
quantum sensing application. The objective is a function of a two sets of
inputs: d^{init} and d^{pert}

Given these, the objective is
sum_{j=1}^m (sqrt(d^{init}_j) - sqrt(d^{init}_j))**2

This is the hfun. So given x, the Ffun must compute/return d^{init} and d^{pert}
"""

import ibcdfo.pounders as pdrs
import numpy as np
from declare_hfun_and_combine_model_with_jax_CFI import hfun, combinemodels_jax


def Ffun(x, nostruct=True):
    # This is a synthetic Ffun. The real example calls an expensive-to-evaluate
    # quantum system to obtain d^{init} and d^{pert}
    number_of_js = 2**len(x)

    # Define d_init as a sinusoidal function of x, scaled by the index
    d_init = np.array([1+np.sin(i + 1 + np.sum(x)) for i in range(number_of_js)])

    # Define d_pert as a polynomial function of x
    d_pert = np.array([np.sum(x)**(2) + (i + 1)**0.5 for i in range(number_of_js)])

    out = np.squeeze(np.hstack((d_init, d_pert)))
    if nostruct:
        return hfun(out)
    else:
        return out


nf_max = 500
g_tol = 1e-5
n = 5
X_0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Low = -10 * np.ones((1, n))
Upp = 10 * np.ones((1, n))
delta = 0.1

hF = {}
for call in range(2):
    if call == 0:
        # Call pounders with m=1 building models of hfun(Ffun(x)) directly (not using structure)
        Ffun_to_use = lambda x: Ffun(x, True)
        m = 1  # not using structure
        Opts = {
            "hfun": lambda F: np.squeeze(F),  # not using structure
            "combinemodels": pdrs.identity_combine,  # not using structure
        }
    elif call == 1:
        # Calls pounders to combine models of Ffun components using the derivatives of hfun (obtained by jax)
        Ffun_to_use = lambda x: Ffun(x, False)
        m = 2 ** (n+1)  # using structure
        Opts = {
            "hfun": hfun,  # using structure
            "combinemodels": combinemodels_jax,  # using structure
        }

    [X, F, hF[call], flag, xkin] = pdrs.pounders(Ffun_to_use, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
    # print(X,xkin,hF[call])
    assert flag == 0, "Didn't reach critical point"


print(f"Using structure uses {len(hF[1])} evals. Not using structure uses {len(hF[0])}")
assert len(hF[1]) < len(hF[0]), "While not true for every problem, using structure on this problem should be beneficial"
