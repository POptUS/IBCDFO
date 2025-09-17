def prepare_outputs_before_return(X, F, h, nf, xkin, exit_flag):
    """
    This function is called to cleanup X and F, set the exit value, and display
    reason for exiting.
    """

    X = X[: nf + 1]
    F = F[: nf + 1]
    h = h[: nf + 1]

    if exit_flag == -2:
        print("MSP: Minimize affine envelope subproblem failed. Problem likely unbounded or poorly scaled.")
    if exit_flag == -1:
        print("MSP: Model building failed. Empty Gres.")
    elif exit_flag == 0:
        print("MSP: Evaluation budget exceeded. Exiting")
    elif exit_flag > 0:
        print("MSP: Trust-region radius less than mindelta. Exiting with chi_k as exit_flag.")

    return X, F, h, xkin, exit_flag
