import numpy as np


def bqmin(A, B, L, U):
    """
    bqmin(A,B,L,U) -> [X,f]
      Minimizes the quadratic 0.5 * X.T @ A @ X + B subject to L<=X<=U using the
      projected gradient method with a (semi) exact line search.
      This will one day be replaced by a more efficient solver.
      This approach is not recommended for n>100.
     --INPUTS-----------------------------------------------------------------
     A       [dbl] [n-by-n] (Symmetric) Hessian matrix format
     B       [dbl] [n-by-1] Gradient vector
     L       [dbl] [1-by-n] Vector of lower bounds assumed to be nonpositive
     U       [dbl] [1-by-n] Vector of upper bounds, must have U(j)>=0>=L(j)
     --OUTPUTS----------------------------------------------------------------
     X       [dbl] [n-by-1] Approximate solution
     f       [dbl] Function value at X
    function [X,f] = bqmin(A,B,L,U)
     --INTERMEDIATE-----------------------------------------------------------
     G       [dbl] [n-by-1]  Gradient at X
     it      [dbl] Iteration counter
     pap     [dbl] The A norm of the projected gradient
     Projg   [dbl] [n-by-1]  Projected gradient at X
     t       [dbl] Step length along projected gradient
    """
    # Internal Parameters
    n = np.shape(A)[1]  # [int] Dimension (number of continuous variables)
    maxit = 5000  # [int] maximum number of iterations
    pgtol = 1e-13  # [dbl] tolerance on final projected gradient
    # Initial point (assumed feasible by L <= 0 <= U)
    X = np.zeros(n)
    f = X.T @ (0.5 * A @ X + B)
    G = A @ X + B
    Projg = X - np.maximum(np.minimum(X - G, U), L)  # Projected gradient
    it = 0  # iteration counter
    while it < maxit and np.linalg.norm(Projg, 2) > pgtol:
        it += 1
        # Simple line search along the projected gradient
        t = 1  # By default take the full step
        pap = Projg.T @ (A @ Projg)
        if pap > 0:
            t = np.minimum(1, (Projg.T @ G) / pap)
        # Compute the next point and update everything
        X = X - t * Projg
        f = X.T @ (0.5 * A @ X + B)
        G = A @ X + B
        Projg = X - np.maximum(np.minimum(X - G, U), L)
    # f has type array([[dbl]])
    # f = f[0][0]
    return [X, f]
