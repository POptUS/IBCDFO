# This tests pounders and MSP against two BenDFO oracles, chosen because their
# output dimension m = 11, which is the case for the FES application being solved. 
#
# NOTE: In the intended application, gamma(kappa, Delta, zeta, KY) depends
#       on only four input parameters (kappa, Delta, zeta, KY).
#       The BenDFO test functions used in this test do not have n = 4,
#       so these tests are only for testing the algorithms and hfun definition
#       rather than representing the true physics problem.
import os

import ibcdfo.pounders as pdrs

# import matplotlib.pyplot as plt
import numpy as np
from calfun import calfun
from dfoxs import dfoxs
from ibcdfo.manifold_sampling.h_examples import max_gamma_over_KY as hfun
from ibcdfo.manifold_sampling.manifold_sampling_primal import manifold_sampling_primal

dfo = np.loadtxt("dfo.dat")

Results = {}
probs_to_solve = [16, 33]

subprob_switch = "linprog"
nfmax = 150

# Make sure plots directory exists
os.makedirs("plots", exist_ok=True)

for row, (nprob, n, m, factor_power) in enumerate(dfo[probs_to_solve, :]):
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, 10**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X, F, h_msp, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)

    # --- Run pounders without using the structure ---
    combinemodels = pdrs.identity_combine

    def unstructured_obj(x):
        maxout = hfun(Ffun(x))
        return np.squeeze(maxout[0])  # only the function value

    identity_hfun = lambda F: np.squeeze(F)

    nf_max = 200
    g_tol = 10**-13
    delta = 0.1

    Opts = {"spsolver": 1, "hfun": identity_hfun, "combinemodels": combinemodels}

    X, F, h_pounders, flag, xk_in = pdrs.pounders(unstructured_obj, x0, n, nf_max, g_tol, delta, 1, LB, UB, Options=Opts)

    # # --- Plotting ---
    # plt.figure(figsize=(10, 4))

    # # Raw values
    # plt.subplot(1, 2, 1)
    # plt.plot(h_pounders, label="pounders (raw)")
    # plt.plot(h_msp, label="msp (raw)")
    # plt.xlabel("Iteration / evaluation")
    # plt.ylabel("h values")
    # plt.title(f"Raw h values (prob {int(nprob)})")
    # plt.legend()

    # # Cumulative minima
    # plt.subplot(1, 2, 2)
    # plt.plot(np.minimum.accumulate(h_pounders), label="pounders (cummin)")
    # plt.plot(np.minimum.accumulate(h_msp), label="msp (cummin)")
    # plt.xlabel("Iteration / evaluation")
    # plt.ylabel("Best-so-far h values")
    # plt.title(f"Cumulative min h values (prob {int(nprob)})")
    # plt.legend()

    # plt.tight_layout()

    # # Save to file
    # fname = os.path.join("plots", f"prob{int(nprob)}.png")
    # plt.savefig(fname, dpi=200)
    # plt.close()
    # print(f"Saved plot for problem {int(nprob)} to {fname}")
