import matplotlib.pyplot as plt
import os
import sys

import numpy as np

sys.path.append("../")
sys.path.append("../../../../BenDFO/py/")
sys.path.append("../h_examples/")
from manifold_sampling_primal import manifold_sampling_primal
from pw_maximum import pw_maximum as hfun

n = 10

np.random.seed(0)
A = {}
for i in range(n):
    # Make 5 random positive-and negative-definite matricies
    B = np.random.uniform(-1, 1, (20, 20))
    A[i] = (-1) ** i * 0.5 * (B @ B.T)


def Ffun(y):
    y = y.squeeze()
    assert len(y) == n, "Wrong input dimension"
    M = np.zeros((20, 20))
    for i, val in enumerate(y):
        M += val * A[i]

    eigvals, eigvecs = np.linalg.eig(M)

    # np.random.shuffle(eigvals)
    eigvals.sort()
    return [np.max(eigvals)]


nfmax = 80
subprob_switch = "linprog"
LB = -1 * np.ones((1, n))
UB = np.ones((1, n))
# x0 = np.array([(-1)**(i) for i in range(n)])
x0 = np.ones((1, n))
# x0 = np.array([(-1)**(i+1) for i in range(10)])
# print(Ffun(x0))
# print(Ffun(x0))

X, F, h, xkin, flag = manifold_sampling_primal(hfun, Ffun, x0, LB, UB, nfmax, subprob_switch)

# plt.plot(F, linewidth=6, alpha=0.8, solid_joinstyle='miter')
plt.plot(F, linewidth=6, alpha=0.8)
plt.savefig("Initial.png", dpi=300, bbox_inches="tight")
plt.close()

# plt.plot(h, linewidth=6, alpha=0.8, solid_joinstyle='miter', label="Getting sorted")
# plt.legend()
# plt.savefig("Initial.png", dpi=300, bbox_inches="tight")
