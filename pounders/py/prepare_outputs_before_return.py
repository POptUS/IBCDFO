def prepare_outputs_before_return(X, F, nf):
    """
    Clean X and F so that they only contain data from actual evaluations.
    """
    return X[:, nf + 1], F[:, nf + 1]
