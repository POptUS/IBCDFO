import sys

import ibcdfo.pounders as pdrs
import numpy as np
from calFun import calFun
from scipy.io import loadmat

# Sample calling syntax for pounders
# func is a function imported from calFun.py as calFun
func = calFun
# n [int] Dimension (number of continuous variables)
n = 16
# X0  [dbl] [min(fstart,1)-by-n] Set of initial points  (zeros(1,n))
np.random.seed(0)
# X0 = 0.5 + 0.1*np.random.rand(1, n)
dataDictionary = loadmat("callpoundersX0.mat")
X0 = dataDictionary["X0"]
# npmax [int] Maximum number of interpolation points (>n+1) (2*n+1)
npmax = 2 * n + 1
# nfmax   [int] Maximum number of function evaluations (>n+1) (100)
nfmax = 200
# gtol [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
gtol = 10**-13
# delta [dbl] Positive trust region radius (.1)
delta = 0.1
# nfs  [int] Number of function values (at X0) known in advance (0)
nfs = 0
# m [int] number of residuals
m = n
# F0 [dbl] [fstart-by-1] Set of known function values  ([])
F0 = []
# xind [int] Index of point in X0 at which to start from (0)
xind = 0
# Low [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
Low = 0.5 * np.ones((1, n))
# Upp [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
Upp = 0.8 * np.ones((1, n))
# printf = True to indicate you want output to screen
printf = False
# Choose your solver:
spsolver = 1
# spsolver=2
[X, F, flag, xkin] = pdrs.pounders(func, X0, n, npmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf)
