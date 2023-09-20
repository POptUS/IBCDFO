import numpy as np

from ibcdfo.pounders import pounders
from ibcdfo.pounders import general_h_funs


def call_beamline_simulation(x):
    # In here, put your call to your simulation that takes in the
    # parameters x and returns the three values used in the calculation of
    # emittance.
    # out = put_your_sim_call_here(x)

    out = x.squeeze()[:3]  # This is not doing any beamline simulation!

    assert len(out) == 3, "Incorrect output dimension"
    return np.squeeze(out)


def doit():
    # Adjust these:
    n = 4  # Nubmer of parameters to be optimized
    X0 = np.random.uniform(0, 1, (1, n))  # starting parameters for the optimizer
    nfmax = int(100)  # Max number of evaluations to be used by optimizer
    Low = -1 * np.ones((1, n))  # 1-by-n Vector of lower bounds
    Upp = np.ones((1, n))  # 1-by-n Vector of upper bounds
    printf = True

    # Don't adjust these:
    hfun = general_h_funs.emittance_h
    combinemodels = general_h_funs.emittance_combine
    m = 3
    gtol = 1e-8
    delta = 0.1
    mpmax = 2 * n + 1  # Maximum number of interpolation points [2*n+1]
    F0 = np.zeros((1, m))
    F0[0] = call_beamline_simulation(X0)
    nfs = 1
    xind = 0

    # The call to the method
    [Xout, Fout, flag, xkinout] = pounders(call_beamline_simulation, X0, n, mpmax, nfmax, gtol, delta, nfs, m, F0, xind, Low, Upp, printf, 1, hfun, combinemodels)

    assert flag != 1, "pounders crashed"

    evals = Fout.shape[0]
    h = np.zeros(evals)

    for i in range(evals):
        h[i] = hfun(Fout[i, :])


if __name__ == "__main__":
    doit()
