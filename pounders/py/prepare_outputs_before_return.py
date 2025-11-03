import poptus


def prepare_outputs_before_return(X, F, hF, nf, logger, exit_flag):
    """
    This function is called to cleanup X and F, set the exit value, and display
    reason for exiting.
    """

    def log(msg):
        logger.log("POUNDers", msg, poptus.LOG_LEVEL_DEFAULT)

    X = X[: nf + 1]
    F = F[: nf + 1]
    hF = hF[: nf + 1]

    if exit_flag == 0:
        log("g is sufficiently small.")
    elif exit_flag == -1:
        log("Number of residuals in output of fun does not match supplied m. Exiting.")
    elif exit_flag == -2:
        log("Terminating because mdec == 0 with a valid model and no improvement from TRSP solution.")
    elif exit_flag == -3:
        log("A NaN was encountered in an objective evaluation. Exiting.")
    elif exit_flag == -4:
        log("A minq input error occurred. Exiting.")
    elif exit_flag == -5:
        log("Unable to improve model with current Pars; try dividing Par[2:3] by 10.")
    elif exit_flag == -6:
        log("Terminating because delta_min reached with a valid model.")

    return X, F, hF, exit_flag
