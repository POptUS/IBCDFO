# This tests the ability of the Python pounders implementation to
# gracefully terminate when receiving a NaN as a component in the vector of
# objective output.

import sys

import ibcdfo.pounders as pdrs
import numpy as np


def failing_objective(x):
    fvec = x

    if np.random.uniform() < 0.1:
        fvec[0] = np.nan

    return fvec


spsolver = 1
nfmax = 1000
gtol = 1e-13
n = 3
m = 3

X0 = np.array([10, 20, 30])
npmax = 2 * n + 1
L = -np.inf * np.ones(n)
U = np.inf * np.ones(n)
nfs = 0
F0 = []
xkin = 0
delta = 0.1
printf = 0

np.random.seed(1)

[X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver)
assert flag == -3, "No NaN was encountered in this test, but should have been."


F0 = np.array([1.0, 2.0])
nfs = 2
[X, F, flag, xk_best] = pdrs.pounders(failing_objective, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xkin, L, U, printf, spsolver)
assert flag == -1, "We are testing proper failure of pounders"
