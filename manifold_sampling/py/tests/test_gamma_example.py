import sys

sys.path.append("./jaxnp_hash/")

import numpy as np

import ibcdfo
from calfun import calfun
from dfoxs import dfoxs
from ibcdfo.manifold_sampling import h_max_gamma_over_KY
from jan_example import h_max_gamma_over_KY_jax

dfo = np.loadtxt("dfo.dat")

PROBS_TO_SOLVE = [16, 33]
SUBPROB_SWITCH = "linprog"
NF_MAX = 150

for prob_row in PROBS_TO_SOLVE:
    nprob, n, m, factor_power = dfo[prob_row, :]
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, 10**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X_old, F_old, h_old, xkin_old, flag_old = ibcdfo.run_MSP(h_max_gamma_over_KY, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
    X_jax, F_jax, h_jax, xkin_jax, flag_jax = ibcdfo.run_MSP(h_max_gamma_over_KY_jax, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)

    assert np.allclose(X_old, X_jax, rtol=1e-6, atol=1e-8), f"X trajectories diverge for dfo row {prob_row}"
    assert np.allclose(F_old, F_jax, rtol=1e-6, atol=1e-8), f"F trajectories diverge for dfo row {prob_row}"
    assert np.allclose(h_old, h_jax, rtol=1e-6, atol=1e-8), f"h_msp trajectories diverge for dfo row {prob_row}"

    print(f"dfo row {prob_row} (prob {int(nprob)}): hand-coded and jax h_max_gamma_over_KY agree")
