import numpy as np
import ibcdfo


def call_beamline_simulation(x):
    # In here, put your call to your simulation that takes in the
    # parameters x and returns the three values used in the calculation of
    # emittance.
    # out = put_your_sim_call_here(x)

    out = x.squeeze()[:3]  # This is not doing any beamline simulation!

    assert len(out) == 3, "Incorrect output dimension"
    return np.squeeze(out)


np.random.seed(8675309)
# Adjust these:
n = 4  # Number of parameters to be optimized
X_0 = np.random.uniform(0, 1, (1, n))  # starting parameters for the optimizer
nf_max = int(100)  # Max number of evaluations to be used by optimizer
Low = -1 * np.ones((1, n))  # 1-by-n Vector of lower bounds
Upp = np.ones((1, n))  # 1-by-n Vector of upper bounds
Ffun = call_beamline_simulation  # Simulation function, accepting single points to evaluate
printf = True

# Not as important to adjust:
hfun = ibcdfo.pounders.h_emittance
combinemodels = ibcdfo.pounders.combine_emittance
m = 3  # The number of outputs from the beamline simulation. Should be 3 for emittance minimization
g_tol = 1e-8  # Stopping tolerance
delta_0 = 0.1  # Initial trust-region radius
F_0 = np.zeros((1, m))  # Initial evaluations (parameters with completed simulations)
F_0[0] = Ffun(X_0)
nfs = 1  # Number of initial evaluations
xk_in = 0  # Index in F_0 for starting the optimization (usually the point with minimal emittance)

Options = {}
Options["printf"] = printf
Options["hfun"] = hfun
Options["combinemodels"] = combinemodels

Prior = {"X_init": X_0, "F_init": F_0, "nfs": nfs, "xk_in": xk_in}

# The call to the method
[Xout, Fout, hFout, flag, xk_inout] = ibcdfo.run_pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Prior=Prior, Options=Options, Model={})

assert flag >= 0, "pounders crashed"

assert hFout[xk_inout] == np.min(hFout), "The minimum emittance is not at xk_inout"
