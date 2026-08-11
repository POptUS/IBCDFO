# Runs manifold sampling on ONE benchmark problem with either the hand-coded or the
# jax-hash hfun, so the two can be timed/profiled in isolation from each other and from
# pytest overhead.
#
# Problem picked: h_one_norm on dfo row 0 (n=9, m=45) -- this is the combo the full
# comparison-plot run (plot_hand_coded_vs_jax_progress.py) stalled on for 35+ minutes
# without finishing a single evaluation, unlike the smaller rows (n<=8, m<=8) it had
# previously been tested on. m=45 means h_one_norm's jax version traces abs() over 45
# components; if jaxnp_hash enumerates tie-break branches combinatorially in the number
# of near-zero components, going from m=5-8 to m=45 could be exponential rather than
# linear -- this script isolates that combo so it can be profiled directly.
#
# Usage:
#   python run_single_problem.py hand
#   python run_single_problem.py jax
import sys
import time

sys.path.append("./jaxnp_hash/")

import numpy as np

import ibcdfo
import ibcdfo.manifold_sampling as ms
import jan_example as je
from calfun import calfun
from dfoxs import dfoxs

if len(sys.argv) != 2 or sys.argv[1] not in ("hand", "jax"):
    sys.exit("Usage: python run_single_problem.py [hand|jax]")

VERSION = sys.argv[1]

DFO_ROW = 0
NAME = "h_one_norm"
SUBPROB_SWITCH = "linprog"
NF_MAX = 150

dfo = np.loadtxt("dfo.dat")
nprob, n, m, factor_power = dfo[DFO_ROW, :]
n = int(n)
m = int(m)
LB = -np.inf * np.ones((1, n))
UB = np.inf * np.ones((1, n))
x0 = dfoxs(n, nprob, 10**factor_power)


def Ffun(y):
    out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
    assert len(out) == m, "Incorrect output dimension"
    return np.squeeze(out)


hfun = ms.h_one_norm if VERSION == "hand" else je.h_one_norm_jax

print(f"Running {NAME} ({VERSION}) on dfo row {DFO_ROW} (prob {int(nprob)}, n={n}, m={m}), nf_max={NF_MAX}")
t0 = time.time()
X, F, h_msp, xkin, flag = ibcdfo.run_MSP(hfun, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s, flag={flag}, evals={len(X)}, best h={np.min(h_msp):.6e}")
