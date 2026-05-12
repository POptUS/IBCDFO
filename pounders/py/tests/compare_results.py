import scipy.io

import numpy as np


def compare_results(filename_benchmark, filename_result):
    """
    Insist on bitwise identical for now.  Tolerances can be added in later if
    they are deemed necessary.
    """
    # ----- HARDCODED VALUES
    EXPECTED_KEYS = {"alg", "problem", "H", "Fvec", "X", "flag", "xk_best"}

    # ----- LOAD FULL RESULTS & CONFIRM SAME PROBLEM
    ref = scipy.io.loadmat(filename_benchmark)
    new = scipy.io.loadmat(filename_result)

    ref_keys = [k for k in ref.keys() if not k.startswith("__")]
    new_keys = [k for k in new.keys() if not k.startswith("__")]
    assert set(ref_keys) == EXPECTED_KEYS
    assert set(ref_keys) == set(new_keys)

    if ref["alg"] != new["alg"]:
        msg = "ERROR: {} and {} used different algorithms ({} != {})"
        print(msg.format(filename_benchmark, filename_result, ref["alg"], new["alg"]))
        return False
    if ref["problem"] != new["problem"]:
        msg = "ERROR: {} and {} used different problems ({} != {})"
        print(msg.format(filename_benchmark, filename_result, ref["problem"], new["problem"]))
        return False

    # ----- LOAD BENCHMARK RESULTS & SANITY CHECK
    H_ref = np.squeeze(ref["H"])
    assert H_ref.ndim == 1
    n_evaluations = len(H_ref)
    assert all(np.isreal(H_ref))
    assert all(np.isfinite(H_ref))

    F_ref = np.squeeze(ref["Fvec"])
    assert F_ref.ndim == 2
    tmp, m = F_ref.shape
    assert tmp == n_evaluations
    assert all(np.isreal(F_ref.flatten()))
    assert all(np.isfinite(F_ref.flatten()))

    X_ref = np.squeeze(ref["X"])
    assert X_ref.ndim == 2
    tmp, n = X_ref.shape
    assert tmp == n_evaluations
    assert all(np.isreal(X_ref.flatten()))
    assert all(np.isfinite(X_ref.flatten()))

    ref_flag = np.squeeze(ref["flag"])
    assert ref_flag.ndim == 0
    ref_flag = float(ref_flag)
    assert np.isreal(ref_flag)
    assert np.isfinite(ref_flag)

    ref_x_best = np.squeeze(ref["xk_best"])
    assert ref_x_best.ndim == 0
    assert ref_x_best - np.floor(ref_x_best) == 0.0
    ref_x_best = int(ref_x_best)

    # ----- COMPARE NEW RESULTS AGAINST BENCHMARK
    H_new = np.squeeze(new["H"])
    if H_new.shape != H_ref.shape:
        msg = "ERROR: H arrays have different shapes ({} != {})"
        print(msg.format(H_ref.shape, H_new.shape))
        return False
    if any(H_new != H_ref):
        max_abs_diff = np.max(np.fabs(H_new - H_ref))
        print(f"ERROR: H absolute differences as large as {max_abs_diff}")
        return False

    F_new = np.squeeze(new["Fvec"])
    if F_new.shape != F_ref.shape:
        msg = "ERROR: Fvec arrays have different shapes ({} != {})"
        print(msg.format(F_ref.shape, F_new.shape))
        return False
    if any(F_new.flatten() != F_ref.flatten()):
        max_abs_diff = np.max(np.fabs(F_new.flatten() - F_ref.flatten()))
        print(f"ERROR: Fvec absolute differences as large as {max_abs_diff}")
        return False

    X_new = np.squeeze(new["X"])
    if X_new.shape != X_ref.shape:
        msg = "ERROR: X arrays have different shapes ({} != {})"
        print(msg.format(X_ref.shape, X_new.shape))
        return False
    if any(X_new.flatten() != X_ref.flatten()):
        max_abs_diff = np.max(np.fabs(X_new.flatten() - X_ref.flatten()))
        print(f"ERROR: X absolute differences as large as {max_abs_diff}")
        return False

    new_flag = np.squeeze(new["flag"])
    assert new_flag.ndim == 0
    new_flag = float(new_flag)
    if new_flag != ref_flag:
        max_abs_diff = np.fabs(new_flag - ref_flag)
        print(f"ERROR: Flag absolute difference = {max_abs_diff}")
        return False

    new_x_best = np.squeeze(new["xk_best"])
    assert new_x_best.ndim == 0
    assert new_x_best - np.floor(new_x_best) == 0.0
    new_x_best = int(new_x_best)
    if new_x_best != ref_x_best:
        msg = "ERROR: Different xk_best integers ({} != {})"
        print(msg.format(ref_x_best, new_x_best))
        return False

    return True
