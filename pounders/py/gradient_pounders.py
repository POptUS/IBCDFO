import sys
import numpy as np
import ipdb
from prepare_outputs_before_return_gradient import prepare_outputs_before_return


def pouders(fun, X0, n, nfmax, gtol, delta, m, L, U, printf=0, spsolver=2, hfun=None, combinemodels=None):
    """
    POUDERS: Practical Optimization Using Derivatives for sums of Squares
      [X,F,flag,xkin] = ...
           pouders(fun,X0,n,npmax,nfmax,gtol,delta,nfs,m,F0,xkin,L,U,printf)

    This code minimizes output from a structured blackbox function, solving
    min { f(X)=sum_(i=1:m) F_i(x)^2, such that L_j <= X_j <= U_j, j=1,...,n }
    where the user-provided blackbox F is specified in the handle fun. Evaluation
    of this F must result in the return of a 1-by-m row vector. Bounds must be
    specified in U and L but can be set to L=-Inf(1,n) and U=Inf(1,n) if the
    unconstrained solution is desired. The algorithm will not evaluate F
    outside of these bounds, but it is possible to take advantage of function
    values at infeasible X if these are passed initially through (X0,F0).
    In each iteration, the algorithm forms an interpolating quadratic model
    of the function and minimizes it in an infinity-norm trust region.

    This software comes with no warranty, is not bug-free, and is not for
    industrial use or public distribution.
    Direct requests and bugs to wild@mcs.anl.gov.
    A technical report/manual is forthcoming, a brief description is in
    Nuclear Energy Density Optimization. Phys. Rev. C, 82:024313, 2010.

    --INPUTS-----------------------------------------------------------------
    fun     [f h] Function handle so that fun(x) evaluates F (@calfun)
    X0      [dbl] [max(nfs,1)-by-n] Set of initial points  (zeros(1,n))
    n       [int] Dimension (number of continuous variables)
    nfmax   [int] Maximum number of function evaluations (>n+1) (100)
    gtol    [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
    delta   [dbl] Positive trust region radius (.1)
    m       [int] Number of residual components
    L       [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
    U       [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
    printf  [log] 0 No printing to screen (default)
                  1 Debugging level of output to screen
                  2 More verbose screen output
    spsolver [int] Trust-region subproblem solver flag (2)

    Optionally, a user can specify and outer-function that maps the the elements
    of F to a scalar value (to be minimized). Doing this also requires a function
    handle (combinemodels) that tells pounders how to map the linear and
    quadratic terms from the residual models into a single quadratic TRSP model.

    hfun           [f h] Function handle for mapping output from F
    combinemodels  [f h] Function handle for combine residual models

    --OUTPUTS----------------------------------------------------------------
    X       [dbl] [nfmax+nfs-by-n] Locations of evaluated points
    F       [dbl] [nfmax+nfs-by-m] Function values of evaluated points
    flag    [dbl] Termination criteria flag:
                  = 0 normal termination because of grad,
                  > 0 exceeded nfmax evals,   flag = norm of grad at final X
                  = -1 if input was fatally incorrect (error message shown)
                  = -2 if a valid model produced X[nf] == X[xkin] or (mdec == 0, Fs[nf] == Fs[xkin])
                  = -3 error if a NaN was encountered
                  = -4 error in TRSP Solver
                  = -5 unable to get model improvement with current parameters
    xkin    [int] Index of point in X representing approximate minimizer
    """
    if hfun is None:

        def hfun(F):
            return np.sum(F**2)

        from general_h_funs import leastsquares as combinemodels

    # choose your spsolver
    if spsolver == 2:
        try:
            from minqsw import minqsw
        except ModuleNotFoundError as e:
            print(e)
            sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

    maxdelta = min(0.5 * np.min(U - L), (10**3) * delta)
    mindelta = min(delta * (10**-13), gtol / 10)
    gam0 = 0.5
    gam1 = 2
    eta1 = 0.05

    eps = np.finfo(float).eps  # Define machine epsilon
    if printf:
        print("  nf   delta       f0           g0       ")
        progstr = "%4i %9.2e  %11.5e %12.4e \n"  # Line-by-line
    X = np.vstack((X0, np.zeros((nfmax - 1, n))))
    F = np.zeros((nfmax, m))
    J = np.zeros((nfmax, n, m))
    Fs = np.zeros(nfmax)
    nf = 0  # in Matlab this is 1
    xkin = 0

    # first evaluation:
    F0, J0 = fun(X[nf])
    F0 = np.atleast_2d(F0)

    if F0.shape[1] != m:
        X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -1)
        if printf:
            print("Your residual is not m-dimensional.")
        return X, F, J, flag, xkin

    if J0.shape[0] != n or J0.shape[1] != m:
        if printf:
            print("Your Jacobian is not n by m.")
        X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -1)
        return X, F, J, flag, xkin

    F[nf] = F0
    J[nf] = J0

    if np.any(np.isnan(F[nf])):
        X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -3)
        return X, F, J, flag, xkin
    if printf:
        print("%4i    Initial point  %11.5e\n" % (nf, hfun(F[nf])))

    # if we had previous evaluations (an nfs ~=0), we would put them in X, F here
    for i in range(nf + 1):
        Fs[i] = hfun(F[i])
    Res = np.zeros(np.shape(F))
    Hres = np.zeros((n, n, m))
    ng = np.nan  # Needed for early termination, e.g., if a model is never built

    while nf + 1 < nfmax:
        #  1a. Compute the "interpolation set".
        Res[xkin] = F[xkin]
        Gres = J[xkin]

        #  1b. Update the quadratic model
        Cres = F[xkin]
        #Hres = Hres + Hresdel
        G, H = combinemodels(Cres, Gres, Hres)
        ind_Lnotbinding = (X[xkin] > L) * (G.T > 0)
        ind_Unotbinding = (X[xkin] < U) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lnotbinding + ind_Unotbinding).T, 2)

        if printf:
            print(progstr % (nf, delta, Fs[xkin], ng))

        # 2. Critically test invoked if the projected model gradient is small
        if ng < gtol:
            X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, 0)
            return X, F, J, flag, xkin

        # 3. Solve the subproblem min{G.T * s + 0.5 * s.T * H * s : Lows <= s <= Upps }
        Lows = np.maximum(L - X[xkin], -delta * np.ones((np.shape(L))))
        Upps = np.minimum(U - X[xkin], delta * np.ones((np.shape(U))))
        if spsolver == 1:  # Stefan's crappy 10line solver
            [Xsp, mdec] = bqmin(H, G, Lows, Upps)
        elif spsolver == 2:  # Arnold Neumaier's minq5
            [Xsp, mdec, minq_err, _] = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
            if minq_err < 0:
                X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -4)
                return X, F, J, flag, xkin
        # elif spsolver == 3:  # Arnold Neumaier's minq8
        #     [Xsp, mdec, minq_err, _] = minq8(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        #     assert minq_err >= 0, "Input error in minq"
        Xsp = Xsp.squeeze()
        step_norm = np.linalg.norm(Xsp, np.inf)

        # 4. Evaluate the function at the new point
        if mdec != 0:
            Xsp = np.minimum(U, np.maximum(L, X[xkin] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (U[i] - Xsp[i] < eps * abs(U[i])) and (U[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = U[i]
                    print("eps project!")
                elif (Xsp[i] - L[i] < eps * abs(L[i])) and (L[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = L[i]
                    print("eps project!")

            nf += 1
            X[nf] = Xsp
            F[nf], J[nf] = fun(X[nf])
            if np.any(np.isnan(F[nf])):
                X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -3)
                return X, F, J, flag, xkin
            Fs[nf] = hfun(F[nf])

            rho = (Fs[nf] - Fs[xkin]) / mdec

            # 4a. Update the center
            if rho > 0:
                # Update model to reflect new center
                xkin = nf  # Change current center

            # 4b. Update the trust-region radius:
            if (rho >= eta1) and (step_norm > 0.75 * delta):
                delta = min(delta * gam1, maxdelta)
            else:
                delta = max(delta * gam0, mindelta)
        else:
            if printf:
                print("Model decrease cannot be found, terminating. ")
            X, F, J, flag = prepare_outputs_before_return(X, F, J, nf, -2)
            return X, F, J, flag, xkin

    if printf:
        print("Number of function evals exceeded")
    flag = ng
    return X, F, J, flag, xkin
