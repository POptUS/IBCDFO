# Plots h_msp (f-value) progress for hand-coded vs jax-hash hfuns on the same
# More-Wild benchmark problems used in test_compare_hand_coded_vs_jax_hfuns.py and
# test_gamma_example.py, so the two tie-breaking conventions (hand-coded's simpler
# branching vs jax's fuller branching near ties) can be compared visually -- e.g. where
# h_one_norm's jax version does extra branching (dfo row 42), does it converge
# faster/slower than the hand-coded version?
#
# Not a test: run directly with `python plot_hand_coded_vs_jax_progress.py`. Images are
# written to tests/plots/.
import os
import sys

sys.path.append("./jaxnp_hash/")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import ibcdfo
import ibcdfo.manifold_sampling as ms
import jan_example as je
from calfun import calfun
from dfoxs import dfoxs

if not os.path.exists("mpc_test_files_smaller_Q"):
    os.system("wget https://web.cels.anl.gov/~jmlarson/mpc_test_files_smaller_Q.zip")
    os.system("unzip mpc_test_files_smaller_Q.zip")

C_L1_loss = np.loadtxt("mpc_test_files_smaller_Q/C_for_benchmark_probs.csv", delimiter=",")
D_L1_loss = np.loadtxt("mpc_test_files_smaller_Q/D_for_benchmark_probs.csv", delimiter=",")
Qzb = sio.loadmat("mpc_test_files_smaller_Q/Q_z_and_b_for_benchmark_problems_normalized_subset.mat")

dfo = np.loadtxt("dfo.dat")

DFO_ROWS = [6, 42]
SUBPROB_SWITCH = "linprog"
NF_MAX = 150

# (name, hand-coded hfun/factory, jax hfun/factory, data needed to build it)
HFUN_PAIRS = [
    ("h_one_norm", ms.h_one_norm, je.h_one_norm_jax, None),
    ("h_pw_maximum", ms.h_pw_maximum, je.h_pw_maximum_jax, None),
    ("h_pw_maximum_squared", ms.h_pw_maximum_squared, je.h_pw_maximum_squared_jax, None),
    ("h_pw_minimum", ms.h_pw_minimum, je.h_pw_minimum_jax, None),
    ("h_pw_minimum_squared", ms.h_pw_minimum_squared, je.h_pw_minimum_squared_jax, None),
    ("h_max_plus_quadratic_violation_penalty", ms.h_max_plus_quadratic_violation_penalty, je.h_max_plus_quadratic_violation_penalty_jax, None),
    ("create_piecewise_quadratic_hfun", ms.create_piecewise_quadratic_hfun, je.create_piecewise_quadratic_hfun_jax, "piecewise_quadratic"),
    ("create_censored_L1_loss_hfun", ms.create_censored_L1_loss_hfun, je.create_censored_L1_loss_hfun_jax, "censored_L1_loss"),
]

# h_max_gamma_over_KY needs m == 11 (its KY grid has 11 entries), so it uses its own
# problem rows rather than DFO_ROWS -- same rows as test_gamma_example.py.
GAMMA_ROWS = [16, 33]


def _build_problem(dfo_row):
    nprob, n, m, factor_power = dfo[dfo_row, :]
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones((1, n))
    UB = np.inf * np.ones((1, n))
    x0 = dfoxs(n, nprob, 10**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    return int(nprob), m, LB, UB, x0, Ffun


def _instantiate(old_hfun, jax_hfun, needs_data, dfo_row, m):
    if needs_data == "piecewise_quadratic":
        Qs = Qzb["Q_mat"][dfo_row, 0]
        zs = Qzb["z_mat"][dfo_row, 0]
        cs = Qzb["b_mat"][dfo_row, 0]
        return old_hfun(Qs, zs, cs), jax_hfun(Qs, zs, cs)
    elif needs_data == "censored_L1_loss":
        ind = np.where((C_L1_loss[:, 0] == dfo_row + 1) & (C_L1_loss[:, 1] == 1))
        C = C_L1_loss[ind, 3 : m + 3]
        D = D_L1_loss[ind, 3 : m + 3]
        return old_hfun(C, D), jax_hfun(C, D)
    else:
        return old_hfun, jax_hfun


def _plot(name, nprob, dfo_row, h_old, h_jax):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(h_old, label="hand-coded")
    plt.plot(h_jax, '--', label="jax-hash")
    plt.xlabel("Evaluation")
    plt.ylabel("h value")
    plt.title(f"{name} (prob {nprob}, row {dfo_row}): raw h")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.minimum.accumulate(h_old), label="hand-coded")
    plt.plot(np.minimum.accumulate(h_jax), '--', label="jax-hash")
    plt.xlabel("Evaluation")
    plt.ylabel("Best-so-far h value")
    plt.title(f"{name} (prob {nprob}, row {dfo_row}): cummin h")
    plt.legend()

    plt.tight_layout()

    fname = os.path.join("plots", f"{name}_row{dfo_row}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved plot for {name}, dfo row {dfo_row} to {fname}")


os.makedirs("plots", exist_ok=True)

for name, old_hfun, jax_hfun, needs_data in HFUN_PAIRS:
    for dfo_row in DFO_ROWS:
        nprob, m, LB, UB, x0, Ffun = _build_problem(dfo_row)
        old, jax_v = _instantiate(old_hfun, jax_hfun, needs_data, dfo_row, m)

        _, _, h_old, _, _ = ibcdfo.run_MSP(old, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
        _, _, h_jax, _, _ = ibcdfo.run_MSP(jax_v, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)

        _plot(name, nprob, dfo_row, h_old, h_jax)

for dfo_row in GAMMA_ROWS:
    nprob, m, LB, UB, x0, Ffun = _build_problem(dfo_row)

    _, _, h_old, _, _ = ibcdfo.run_MSP(ms.h_max_gamma_over_KY, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
    _, _, h_jax, _, _ = ibcdfo.run_MSP(je.h_max_gamma_over_KY_jax, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)

    _plot("h_max_gamma_over_KY", nprob, dfo_row, h_old, h_jax)
