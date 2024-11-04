from gradient_pounders import pouders
import numpy as np
import ipdb

def test_rosenbrock(x, a, b):

    F = np.zeros(2)
    J = np.zeros((2, 2))
    F[0] = a - x[0]
    F[1] = b * (x[1] - x[0]**2)

    # notice that the ith COLUMN corresponds to the ith component gradient in J
    J[0, 0] = -1
    J[0, 1] = -2 * b * x[0]
    J[1, 1] = b

    return F, J


# specify function
a, b = 1, 10
fun = lambda x: test_rosenbrock(x, a, b)

# initial point (and dimension n and residual count m)
n = 2
m = 2
X0 = -1 * np.ones(n)

# a budget
nfmax = 100 * (n + 1)

# gradient tolerance
gtol = 1e-13

# initial TR radius
delta = 1.0

# bound constraints
L = -np.inf * np.ones(n)
U = np.inf * np.ones(n)

# note that hfun and combinemodels are least squares BY DEFAULT (hence the Nones)
X, F, J, flag, xkin = pouders(fun, X0, n, nfmax, gtol, delta, m, L, U, printf=True, spsolver=2, hfun=None, combinemodels=None)