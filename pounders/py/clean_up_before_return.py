# This function is called to cleanup X and F, set the exit value, and display
# reason for exiting

# flag    [dbl] Termination criteria flag:
#               = 0 normal termination because of grad,
#               > 0 exceeded nfmax evals,   flag = norm of grad at final X
#               = -1 if input was fatally incorrect (error message shown)
#               = -2 model failure
#               = -3 error from TRSP Solver or if a NaN was encountered
#               = -4 error in TRSP Solver


def clean_up_before_return(X, F, nf, exit_flag):
    X = X[: nf + 1]
    F = F[: nf + 1]

    if exit_flag == -4:
        print("An occured in minq. Exiting.")
    elif exit_flag == -3:
        print("A NaN was encountered in an objective evaluation. Exiting.")
    elif exit_flag == -2:
        print('Terminating because mdec == 0 with a valid model and no improvement from TRSP solution')
    elif exit_flag == 0:
        print('g is sufficiently small')

    return X, F, exit_flag
