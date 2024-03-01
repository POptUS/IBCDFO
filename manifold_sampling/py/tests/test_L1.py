import itertools
import numpy as np
from ibcdfo.manifold_sampling.h_examples import one_norm as hfun
from ibcdfo.manifold_sampling.manifold_sampling_primal import manifold_sampling_primal


def Ffun(x):
    # Rosenbrock
    return [10 * (x[1] - x[0] * x[0]), 1 - x[0]]


n = 2
m = 2

nfmax = 1000
subprob_switch = "linprog"
LB = -2 * np.ones((1, n))
UB = 2 * np.ones((1, n))
x0 = np.array([-1.2, 1.0])

X, F, h, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)
