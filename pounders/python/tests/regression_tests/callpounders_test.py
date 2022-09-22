import numpy as np
from calFun import calFun

import sys

sys.path.append("../../")
from pounders import pounders


# Sample calling syntax for pounders
# func is a function imported from calFun.py as calFun
func = calFun
# n [int] Dimension (number of continuous variables)
n = 2
# X0 [dbl] [min(fstart,1)-by-n] Set of initial points  (zeros(1,n))
X0 = np.zeros((10, 2))
X0[0, :] = 0.5 * np.ones((1, 2))
mpmax = int(0.5 * (n + 1) * (n + 2))
# nfmax [int] Maximum number of function evaluations (>n+1) (100)
nfmax = 60
# gtol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
gtol = 10**-13
# delta [dbl] Positive trust region radius (.1)
delta = 0.1
# nfs [int] Number of function values (at X0) known in advance (0)
nfs = 10
# m [int] number of residuals
m = 2
# F0 [dbl] [fstart-by-1] Set of known function values  ([])
F0 = np.zeros((10, 2))
# xind [int] Index of point in X0 at which to start from (1)
xind = 0
# Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
Low = np.zeros((1, n))
# Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
Upp = np.ones((1, n))
# printf [log] 1 Indicates you want output to screen (1)
printf = True
spsolver = 1

np.random.seed(1)
F0[0, :] = func(X0[0, :])
for i in range(1, 10):
    X0[i, :] = X0[0, :] + 0.2 * np.random.rand(1, 2) - 0.1
    F0[i, :] = func(X0[i, :])

[X, F, flag, xkin] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver)
