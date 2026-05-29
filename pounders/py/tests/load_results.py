import scipy.io

import numpy as np


def load_results(filename):
    """
    Based on benchmark results' file format established at commit
    * fb33cdfd for POUNDERS/Python and
    * 2b4d9603 for POUNDERS/MATLAB.
    """
    EXPECTED_KEYS = {"alg", "problem", "H", "Fvec", "X", "flag", "xk_best"}

    # When adapting this to loading from MATLAB-generated files on GCE, I was
    # forced to add the simplify_cells argument.  Without this, some loaded
    # results had a "W" or a "V" field instead of the "X" field.  When I loaded
    # those same files manually in Python using the exact same venv and
    # scipy.io.loadmat(filename), I always saw the "X" field and didn't see the
    # others.  I don't understand why this was happening nor why the addition
    # of the flag solves it.
    data = scipy.io.loadmat(filename, simplify_cells=True)
    keys = [k for k in data.keys() if not k.startswith("__")]
    assert set(keys) == EXPECTED_KEYS

    algorithm = str(np.squeeze(data["alg"]))

    problem = str(np.squeeze(data["problem"]))
    if (not problem.startswith("problem")) or (not problem.endswith("from More/Wild")):
        raise ValueError(f"Invalid problem spec ({problem})")
    try:
        int(problem.lstrip("problem").rstrip("from More/Wild"))
    except Exception:
        raise ValueError(f"Invalid problem spec ({problem})")

    H = np.squeeze(data["H"])
    assert H.ndim == 1
    n_evaluations = len(H)
    assert all(np.isreal(H))

    # Fvec could be a scalar at each evaluation
    Fvec = np.atleast_2d(np.squeeze(data["Fvec"]))
    assert Fvec.ndim == 2
    tmp, _ = Fvec.shape
    assert tmp == n_evaluations
    assert all(np.isreal(Fvec.flatten()))

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
    if algorithm == "POUNDERS_M":
        # The MATLAB implementation's test suite saves the index of the best
        # approximation as a 1-based integer.  However, we need to adjust it to
        # 0-based since we are returning Python arrays.
        xk_best -= 1
    elif algorithm != "POUNDERS_Py":
        raise ValueError(f"Unknown POUNDERS test algorithm {algorithm}")
    assert xk_best in range(0, len(H))

    return algorithm, problem, X, Fvec, H, xk_best, flag
