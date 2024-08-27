def prepare_outputs_before_return(X, F, hF, nf, exit_flag, Xtype, Group):
    """
    This function is called to cleanup X and F, set the exit value, and display
    reason for exiting.
    """
    X = X[: nf + 1]
    F = F[: nf + 1]
    hF = hF[: nf + 1]
    Xtype = Xtype[: nf + 1]
    Group = Group[: nf + 1]

    if exit_flag == -4:
        print("A minq input error occurred. Exiting.")
    elif exit_flag == -3:
        print("A NaN was encountered in an objective evaluation. Exiting.")
    elif exit_flag == -2:
        print("Terminating because mdec == 0 with a valid model and no improvement from TRSP solution")
    elif exit_flag == -5:
        print("Unable to improve model with current Pars; try dividing Par[2:3] by 10")
    elif exit_flag == -1:
        print("Number of residuals in output of fun does not match supplied m. Exiting.")
    elif exit_flag == 0:
        print("g is sufficiently small")

    return X, F, hF, exit_flag, Xtype, Group
