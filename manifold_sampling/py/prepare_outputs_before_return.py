def prepare_outputs_before_return(X, F, h, nf, exit_flag):
    """
    This function is called to cleanup X and F, set the exit value, and display
    reason for exiting.
    """

    X = X[: nf + 1]
    F = F[: nf + 1]
    h = h[: nf + 1]

    return X, F, h, exit_flag
