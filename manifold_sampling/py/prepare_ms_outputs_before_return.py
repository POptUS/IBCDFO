def prepare_ms_outputs_before_return(X, F, hF, nf, xk_in, exit_flag, printf=1, chi_k=None):
    """
    This function is called to cleanup X and F, set the exit value, and display
    reason for exiting.

    The exit_flag follows the pounders convention: 0 is normal termination, 1
    means the budget was exhausted, -1 is a model-building failure, -2 is a
    subproblem failure, and -6 means mindelta was reached without satisfying
    gtol. A flag of -6 is a common way for a successful run to end, so the
    final stationary measure chi_k, if provided, is displayed to help judge
    the result.

    No message is displayed if printf is 0.
    """

    X = X[: nf + 1]
    F = F[: nf + 1]
    hF = hF[: nf + 1]

    if printf:
        if exit_flag == 0:
            print("MSP: Convergence satisfied: small stationary measure and small delta.")
        elif exit_flag == 1:
            print("MSP: Evaluation budget exceeded. Exiting.")
        elif exit_flag == -1:
            print("MSP: Model building failed. Empty Gres.")
        elif exit_flag == -2:
            print("MSP: Minimize affine envelope subproblem failed. Problem likely unbounded or poorly scaled.")
        elif exit_flag == -6:
            msg = "MSP: Terminating because delta_min reached without satisfying g_tol."
            if chi_k is not None:
                msg += " Final chi_k = %.3e" % chi_k
            print(msg)

    return X, F, hF, xk_in, exit_flag
