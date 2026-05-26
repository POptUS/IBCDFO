import scipy.io

import numpy as np

from .load_results import load_results


def _load_results_py_v2(filename):
    """
    POUNDERS/Python v2 format established at commit 9f5a5f37c
    """
    EXPECTED_KEYS = {"alg", "problem", "H", "Fvec", "X", "flag", "xk_best"}

    contents = scipy.io.loadmat(filename)
    keys = [k for k in contents.keys() if not k.startswith("__")]
    assert len(keys) == 1
    assert keys[0].startswith("pounders4py_")
    data = contents[keys[0]]
    assert set(data.dtype.names) == EXPECTED_KEYS

    algorithm = data["alg"][0][0][0]
    problem = data["problem"][0][0][0]

    H = np.squeeze(data["H"][0][0])
    assert H.ndim == 1
    n_evaluations = len(H)
    assert all(np.isreal(H))
    assert all(np.isfinite(H))

    Fvec = np.squeeze(data["Fvec"][0][0])
    assert Fvec.ndim == 2
    tmp, _ = Fvec.shape
    assert tmp == n_evaluations
    assert all(np.isreal(Fvec.flatten()))
    assert all(np.isfinite(Fvec.flatten()))

    X = np.squeeze(data["X"][0][0])
    assert X.ndim == 2
    tmp, _ = X.shape
    assert tmp == n_evaluations
    assert all(np.isreal(X.flatten()))
    assert all(np.isfinite(X.flatten()))

    flag = np.squeeze(data["flag"][0][0])
    assert np.isreal(flag)
    assert np.isfinite(flag)
    xk_best = np.squeeze(data["xk_best"][0][0])
    assert xk_best in range(0, len(H))

    return algorithm, problem, X, Fvec, H, xk_best, flag


def _load_results_m_v1(filename):
    """
    POUNDERS/MATLAB v1 format established at commit 360d6e29
    """
    raise NotImplementedError("Pending task")


def compare_results(filename_benchmark, filename_result):
    """
    .. todo::
        * Allow for users to specify nonzero tolerances if the use case arises.
        * Allow for checking Python and MATLAB results on a set of problems on
          which we expect all optimizations to find the same local minimizer.
          This would require nonzero tolerances.

    :param filename_benchmark: Filename of |pounders| ``.mat``-format
        benchmarking result that calling code considers to be the accepted
        reference.
    :param filename_result: Filename of |pounders| ``.mat``-format benchmarking
        result that calling code wishes to check against the reference.
    :return: True if the files correspond to identical test setups and contain
        bitwise-identical results.
    """
    # ----- HARDCODED VALUES
    RED = "\033[0;91;1m"  # Bright Red/bold
    BLUE = "\033[0;34;1m"  # Blue/bold
    NC = "\033[0m"  # No Color/Not bold

    # ----- CONSISTENT CLEAN LOGGING OF ERRORS
    def error(msg):
        print(f"{RED}FAIL{NC}\n\t{msg}")

    # ----- LOAD FULL RESULTS & CONFIRM SAME PROBLEM
    print(f"{filename_benchmark.stem} ... ", end="")

    if filename_benchmark.name != filename_result.name:
        error(f"New result has different filename ({filename_result.stem})")
        return False

    ref_alg, ref_problem, X_ref, F_ref, H_ref, x_best_ref, flag_ref = _load_results_py_v2(filename_benchmark)
    new_alg, new_problem, X_new, F_new, H_new, x_best_new, flag_new = load_results(filename_result)

    if ref_alg not in ["pounders4py"]:
        error(f"Invalid algorithm name ({ref_alg}) for benchmark")
        return False
    elif new_alg != "POUNDERS_Py":
        msg = "Benchmark and new result used different algorithms ({} != {})"
        error(msg.format(ref_alg, new_alg))
        return False
    # elif new_alg != ref_alg:
    #     msg = "Benchmark and new result used different algorithms ({} != {})"
    #     error(msg.format(ref_alg, new_alg))
    #     return False

    if (not ref_problem.startswith("problem")) or (not ref_problem.endswith("from More/Wild")):
        error(f"Invalid problem spec ({ref_problem}) for benchmark")
        return False
    try:
        int(ref_problem.lstrip("problem").rstrip("from More/Wild"))
    except Exception:
        error(f"Invalid problem spec ({ref_problem}) for benchmark")
        return False
    if new_problem != ref_problem:
        msg = "Benchmark and new result solve different problems ({} != {})"
        error(msg.format(ref_problem, new_problem))
        return False

    # ----- COMPARE NEW RESULTS AGAINST BENCHMARK
    if len(H_new) != len(H_ref):
        error(f"H arrays have different lengths ({len(H_ref)} != {len(H_new)})")
        return False
    assert F_new.shape == F_ref.shape
    assert X_new.shape == X_ref.shape

    # Don't fail immediately if values are different so that we can provide
    # users with all such differences in one go.
    msgs = []
    if x_best_new != x_best_ref:
        msgs += [f"Best solution indices differ ({x_best_new} != {x_best_ref})"]
    if flag_new != flag_ref:
        msgs += [f"Flags differ ({flag_new} != {flag_ref})"]
    if (flag_new >= 0) and (flag_ref >= 0):
        # Only show comparison if both ran without a hard failure.  For
        # instance, I would like to see the these comparisons if one or both
        # were simply nonconvergent.
        X_best_ref = X_ref[x_best_ref]
        F_best_ref = F_ref[x_best_ref]
        H_best_ref = H_ref[x_best_ref]

        X_best_new = X_new[x_best_new]
        F_best_new = F_new[x_best_new]
        H_best_new = H_new[x_best_new]

        if H_best_new != H_best_ref:
            abs_diff = np.fabs(H_best_new - H_best_ref)
            msgs += [f"H absolute difference = {abs_diff}"]
        if any(F_best_new != F_best_ref):
            max_abs_diff = np.max(np.fabs(F_best_new - F_best_ref))
            msgs += [f"Fvec max absolute difference = {max_abs_diff}"]
        if any(X_best_new != X_best_ref):
            max_abs_diff = np.max(np.fabs(X_best_new - X_best_ref))
            msgs += [f"X max absolute difference = {max_abs_diff}"]

    if msgs:
        error("\n\t".join(msgs))
        return False

    print(f"{BLUE}PASS{NC}")
    return True
