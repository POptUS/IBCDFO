import ibcdfo
import numpy as np

from ibcdfo.pounders._run_user_friendly import run_user_friendly


def call_beamline_simulation(x):
    # In here, put your call to your simulation that takes in the
    # parameters x and returns the three values used in the calculation of
    # emittance.
    # out = put_your_sim_call_here(x)

    out = x.squeeze()[:3]  # This is not doing any beamline simulation!

    assert len(out) == 3, "Incorrect output dimension"
    return np.squeeze(out)


rng = np.random.default_rng(8675309)
# Adjust these:
n = 4  # Number of parameters to be optimized
X_0 = rng.uniform(0, 1, size=n)  # starting parameters for the optimizer
nf_max = int(100)  # Max number of evaluations to be used by optimizer
Low = -1 * np.ones(n)  # 1-by-n Vector of lower bounds
Upp = np.ones(n)  # 1-by-n Vector of upper bounds
Ffun = call_beamline_simulation  # Simulation function, accepting single points to evaluate

# Not as important to adjust:
m = 3  # The number of outputs from the beamline simulation. Should be 3 for emittance minimization
g_tol = 1e-8  # Stopping tolerance
delta_0 = 0.1  # Initial trust-region radius

ObjOpts = {
    "hfun": ibcdfo.pounders.h_emittance,
    "combinemodels": ibcdfo.pounders.combine_emittance,
}
Prior = {
    "nfs": 1,
    "X_init": np.atleast_2d(X_0),
    "F_init": np.atleast_2d(Ffun(X_0)),
    "xk_in": 0,
}

# The call to the method
[Xout, Fout, hFout, flag, xk_inout] = run_user_friendly(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, ObjOpts=ObjOpts, Prior=Prior)

assert flag >= 0, "pounders crashed"

assert hFout[xk_inout] == np.min(hFout), "The minimum emittance is not at xk_inout"
