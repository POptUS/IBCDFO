import numpy as np

from .load_results import load_results


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

    ref_alg, ref_problem, X_ref, F_ref, H_ref, x_best_ref, flag_ref = load_results(filename_benchmark)
    if not all(np.isfinite(H_ref)):
        error("Non-finite h values in benchmark")
        return False
    elif not all(np.isfinite(F_ref.flatten())):
        error("Non-finite Fvec values in benchmark")
        return False

    new_alg, new_problem, X_new, F_new, H_new, x_best_new, flag_new = load_results(filename_result)
    if not all(np.isfinite(H_new)):
        error("Non-finite h values in new results")
        return False
    elif not all(np.isfinite(F_new.flatten())):
        error("Non-finite Fvec values in new results")
        return False

    if new_problem != ref_problem:
        msg = "Benchmark and new result solve different problems ({} != {})"
        error(msg.format(ref_problem, new_problem))
        return False

    # ----- COMPARE NEW RESULTS AGAINST BENCHMARK
    assert F_new.shape[1] == F_ref.shape[1]
    assert X_new.shape[1] == X_ref.shape[1]

    # Don't fail immediately if values are different so that we can provide
    # users with all such differences in one go.
    msgs = []
    if x_best_new != x_best_ref:
        msgs += [f"Best approximation indices differ ({x_best_new} != {x_best_ref})"]
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
    else:
        if flag_ref < 0:
            msgs += [f"Benchmark failed with flag={flag_ref}"]
        if flag_new < 0:
            msgs += [f"New result failed with flag={flag_new}"]

    if msgs:
        error("\n\t".join(msgs))
        return False

    print(f"{BLUE}PASS{NC}")
    return True
