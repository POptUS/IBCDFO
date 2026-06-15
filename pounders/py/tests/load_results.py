import copy

import scipy.io

import numpy as np


def load_results(filename):
    """
    Based on benchmark results' file format established at commit
    * fb33cdfd for POUNDERS/Python and
    * 2b4d9603 for POUNDERS/MATLAB.
    """
    EXPECTED_KEYS = {"alg", "problem", "H", "Fvec", "X", "flag", "xk_best"}

    with open(filename, "rb") as fptr:
        # If I call this function twice in a row on the same MATLAB-generated
        # file (using both scipy v1.16.2 and v1.17.1), on some occasions the "X"
        # field will appear with a different name (e.g., "W") on the second
        # call.  This is apparently related to the change to xk_best below.  If
        # I rerun that same test after exiting Python, I get the same result,
        # which implies that the file itself is not being altered.  I hope that
        # this is the case since we load in readonly mode.  This even happens if
        # the two files loaded back-to-back are different.
        #
        # Therefore, I suspect that there is some sort of caching of or
        # accidental referencing to loaded data.  Caching does not make sense
        # when loading two different files.  However, using a deepcopy resolves
        # the issue.  A simple copy is not sufficient, which makes sense since
        # data is a nested dictionary.
        data = copy.deepcopy(scipy.io.loadmat(fptr))
    keys = [k for k in data.keys() if not k.startswith("__")]
    assert set(keys) == EXPECTED_KEYS

    algorithm = str(np.squeeze(data["alg"]))
    assert algorithm in ["POUNDERS_M", "POUNDERS_Py"]

    problem = str(np.squeeze(data["problem"]))
    if (not problem.startswith("problem")) or (not problem.endswith("from More/Wild")):
        raise ValueError(f"Invalid problem spec ({problem})")
    try:
        int(problem.lstrip("problem").rstrip("from More/Wild"))
    except Exception:
        raise ValueError(f"Invalid problem spec ({problem})")

    H = np.atleast_1d(np.squeeze(data["H"]))
    assert H.ndim == 1
    n_evaluations = len(H)
    assert all(np.isreal(H))
    # Non-finite can happen in our tests, so checking finiteness of H must be
    # handled by calling code.

    # Fvec could be a scalar at each evaluation
    Fvec = np.atleast_2d(data["Fvec"])
    assert Fvec.ndim == 2
    tmp, _ = Fvec.shape
    assert tmp == n_evaluations
    assert all(np.isreal(Fvec.flatten()))
    # Non-finite can happen in our tests, so checking finiteness of Fvec must
    # be handled by calling code.

    # X could be a scalar at each evaluation
    X = np.atleast_2d(data["X"])
    assert X.ndim == 2
    tmp, _ = X.shape
    assert tmp == n_evaluations
    assert all(np.isreal(X.flatten()))
    assert all(np.isfinite(X.flatten()))

    flag = np.squeeze(data["flag"])
    assert (flag >= 0.0) or (flag in [-6, -5, -4, -3, -2, -1])

    xk_best = np.squeeze(data["xk_best"])
    if algorithm == "POUNDERS_M":
        # The MATLAB implementation's test suite saves the index of the best
        # approximation as a 1-based integer.  However, we need to adjust it to
        # 0-based since we are returning Python arrays.
        xk_best -= 1
    assert xk_best in range(0, n_evaluations)

    return algorithm, problem, X, Fvec, H, xk_best, flag
