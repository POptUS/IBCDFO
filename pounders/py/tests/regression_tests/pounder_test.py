# Tests pounders on an m=1 function

import sys

import numpy as np

sys.path.append("../../")
from general_h_funs import identity_combine as combinemodels
from pounders import pounders

# Sample calling syntax for pounders
func = lambda x: np.sum(x)
n = 16

X0 = np.ones(n)
# mpmax [int] Maximum number of interpolation points (>n+1) (2*n+1)
mpmax = 2 * n + 1
# nfmax   [int] Maximum number of function evaluations (>n+1) (100)
nfmax = 200
# gtol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
gtol = 10**-13
# delta [dbl] Positive trust region radius (.1)
delta = 0.1
# nfs  [int] Number of function values (at X0) known in advance (0)
nfs = 0
# m [int] number of residuals
m = 1
# F0 [dbl] [fstart-by-1] Set of known function values  ([])
F0 = []
# xind [int] Index of point in X0 at which to start from (0)
xind = 0
# Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
Low = -0.1 * np.arange(n)
# Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
Upp = np.inf * np.ones(n)
# printf = True to indicate you want output to screen
printf = False
# Choose your solver:
spsolver = 1

hfun = lambda F: F

[X, F, flag, xkin] = pounders(func, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, spsolver, hfun, combinemodels)

assert np.all(X[xkin] == Low), "The optimum show be the lower bounds."
