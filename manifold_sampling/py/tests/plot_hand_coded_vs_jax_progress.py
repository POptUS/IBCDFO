# Plots h_msp (f-value) progress for hand-coded vs jax-hash hfuns on the same
# More-Wild benchmark problems used in test_compare_hand_coded_vs_jax_hfuns.py and
# test_gamma_example.py, so the two tie-breaking conventions (hand-coded's simpler
# branching vs jax's fuller branching near ties) can be compared visually -- e.g. where
# h_one_norm's jax version does extra branching (dfo row 42), does it converge
# faster/slower than the hand-coded version?
#
# DFO_ROWS/HFUN_PAIRS/etc. are imported directly from test_compare_hand_coded_vs_jax_hfuns.py
# rather than duplicated here, so this script always covers whatever problems that test
# currently exercises.
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

import ibcdfo
import ibcdfo.manifold_sampling as ms
import jan_example as je
import test_compare_hand_coded_vs_jax_hfuns as tc

DFO_ROWS = tc.DFO_ROWS
HFUN_PAIRS = tc.HFUN_PAIRS
SUBPROB_SWITCH = tc.SUBPROB_SWITCH
dfo = tc.dfo

# More evals than the fast NF_MAX=20 used by the test, so the plots show real convergence.
NF_MAX = 150

# h_max_gamma_over_KY needs m == 11 (its KY grid has 11 entries), so it uses its own
# problem rows rather than DFO_ROWS -- same rows as test_gamma_example.py.
GAMMA_ROWS = [16, 33]


def _plot(name, nprob, dfo_row, h_old, h_jax):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(h_old, label="hand-coded")
    plt.plot(h_jax, "--", label="jax-hash")
    plt.xlabel("Evaluation")
    plt.ylabel("h value")
    plt.title(f"{name} (prob {nprob}, row {dfo_row}): raw h")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.minimum.accumulate(h_old), label="hand-coded")
    plt.plot(np.minimum.accumulate(h_jax), "--", label="jax-hash")
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
        m, LB, UB, x0, Ffun = tc._build_problem(dfo_row)
        old, jax_v = tc._instantiate(old_hfun, jax_hfun, needs_data, dfo_row, m)
        nprob = int(dfo[dfo_row, 0])

        _, _, h_old, _, _ = ibcdfo.run_MSP(old, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
        _, _, h_jax, _, _ = ibcdfo.run_MSP(jax_v, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)

        _plot(name, nprob, dfo_row, h_old, h_jax)

for dfo_row in GAMMA_ROWS:
    m, LB, UB, x0, Ffun = tc._build_problem(dfo_row)
    nprob = int(dfo[dfo_row, 0])

    _, _, h_old, _, _ = ibcdfo.run_MSP(ms.h_max_gamma_over_KY, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
    _, _, h_jax, _, _ = ibcdfo.run_MSP(je.h_max_gamma_over_KY_jax, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)

    _plot("h_max_gamma_over_KY", nprob, dfo_row, h_old, h_jax)
