import scipy.io

import numpy as np


def load_results(filename):
    """
    POUNDERS/Python v3 format established at commit XXX
    """
    EXPECTED_KEYS = {"alg", "problem", "H", "Fvec", "X", "flag", "xk_best"}

    data = scipy.io.loadmat(filename)
    keys = [k for k in data.keys() if not k.startswith("__")]
    assert set(keys) == EXPECTED_KEYS

    algorithm = np.squeeze(data["alg"])
    problem = np.squeeze(data["problem"])

    H = np.squeeze(data["H"])
    assert H.ndim == 1
    n_evaluations = len(H)
    assert all(np.isreal(H))
    assert all(np.isfinite(H))

    # Fvec could be a scalar at each evaluation
    Fvec = np.atleast_2d(np.squeeze(data["Fvec"]))
    assert Fvec.ndim == 2
    tmp, _ = Fvec.shape
    assert tmp == n_evaluations
    assert all(np.isreal(Fvec.flatten()))
    assert all(np.isfinite(Fvec.flatten()))

    # X could be a scalar at each evaluation
    X = np.atleast_2d(np.squeeze(data["X"]))
    assert X.ndim == 2
    tmp, _ = X.shape
    assert tmp == n_evaluations
    assert all(np.isreal(X.flatten()))
    assert all(np.isfinite(X.flatten()))

    flag = np.squeeze(data["flag"])
    assert np.isreal(flag)
    assert np.isfinite(flag)
    xk_best = np.squeeze(data["xk_best"])
    assert xk_best in range(0, len(H))

    return algorithm, problem, X, Fvec, H, xk_best, flag
