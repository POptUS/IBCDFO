import numpy as np
from ibcdfo.pounders import general_h_funs, pounders


def call_beamline_simulation(x):
    # In here, put your call to your simulation that takes in the
    # parameters x and returns the three values used in the calculation of
    # emittance.
    # out = put_your_sim_call_here(x)

    out = x.squeeze()[:3]  # This is not doing any beamline simulation!

    assert len(out) == 3, "Incorrect output dimension"
    return np.squeeze(out)


# Adjust these:
n = 4  # Number of parameters to be optimized
X0 = np.random.uniform(0, 1, (1, n))  # starting parameters for the optimizer
nfmax = int(100)  # Max number of evaluations to be used by optimizer
Low = -1 * np.ones((1, n))  # 1-by-n Vector of lower bounds
Upp = np.ones((1, n))  # 1-by-n Vector of upper bounds
printf = True

# Not as important to adjust:
hfun = general_h_funs.emittance_h
combinemodels = general_h_funs.emittance_combine
m = 3  # The number of outputs from the beamline simulation. Should be 3 for emittance minimization
gtol = 1e-8  # Stopping tolerance
delta = 0.1  # Initial trust-region radius
mpmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
F0 = np.zeros((1, m))  # Initial evaluations (parameters with completed simulations)
F0[0] = call_beamline_simulation(X0)
nfs = 1  # Number of initial evaluations
xind = 0  # Index in F0 for starting the optimization (usually the point with minimal emittance)

# The call to the method
[Xout, Fout, flag, xkinout] = pounders(call_beamline_simulation, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, 1, hfun, combinemodels)

assert flag != 1, "pounders crashed"

h = np.zeros(Fout.shape[0])

# Compute the emittance values for inspection after optimization
for i in range(len(h)):
    h[i] = hfun(Fout[i, :])

assert h[xkinout] == np.min(h), "The minimum emittance is not at xkinout"
