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


X_0 = np.random.seed(8675309)
# Adjust these:
n = 4  # Number of parameters to be optimized
X_0 = np.random.uniform(0, 1, (1, n))  # starting parameters for the optimizer
nf_max = int(100)  # Max number of evaluations to be used by optimizer
Low = -1 * np.ones((1, n))  # 1-by-n Vector of lower bounds
Upp = np.ones((1, n))  # 1-by-n Vector of upper bounds
printf = True

# Not as important to adjust:
hfun = general_h_funs.emittance_h
combinemodels = general_h_funs.emittance_combine
m = 3  # The number of outputs from the beamline simulation. Should be 3 for emittance minimization
g_tol = 1e-8  # Stopping tolerance
delta_0 = 0.1  # Initial trust-region radius
F_0 = np.zeros((1, m))  # Initial evaluations (parameters with completed simulations)
F_0[0] = call_beamline_simulation(X_0)
nfs = 1  # Number of initial evaluations
xk_in = 0  # Index in F_0 for starting the optimization (usually the point with minimal emittance)

Options = {}
Options["printf"] = printf
Options["hfun"] = hfun
Options["combinemodels"] = combinemodels

Prior = {"X_init": X_0, "F_init": F_0, "nfs": nfs, "xk_in": xk_in}

# The call to the method
[Xout, Fout, hFout, flag, xk_inout] = pounders(call_beamline_simulation, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior=Prior, Options=Options, Model={})

assert flag >= 0, "pounders crashed"

assert hFout[xk_inout] == np.min(hFout), "The minimum emittance is not at xk_inout"
