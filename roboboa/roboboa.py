# This function specifically implements a ROBOBOA algorithm for bilevel problems of the form
# min_{x\in X} max_{u \in U} f(x+u) (*).
# This formulation implies that x and u are vectors of parameters with the same length.
# Moreover, this first attempt will assume that the compact constraints are defined by bounds only, i.e.,
# U is described by U = {u: low <= u <= upp} where <= is understood entrywise.
# This also implies that most good formulations probably satisfy 0\in U.

import numpy as np
from ibcdfo.manifold_sampling.h_examples import pw_maximum
from ibcdfo.manifold_sampling.manifold_sampling_primal_with_gradients import manifold_sampling_primal_with_gradients
import nlopt
import ipdb

def roboboa(funx, x0, b_low, b_upp, nfmax, funxUhat=None, Uhat0=None):

    # Inputs:
    # funx: function that implements f in (*). It should take in an length n numpy array x and return:
    # a scalar f(x) and,
    # a length n numpy array containing the gradient of f at x.
    #
    # x0: an initial guess of optimal parameters, specified as a length n array
    #
    # b_low, b_upp: respectively lower and upper bounds that define U, each specified as a length n array
    #
    # funxUhat: function that implements f in (*), except its first argument is x (as in funx) and its second argument is
    # Uhat, a p x n array. funXU should effectively evaluate fun at x + u_i for i = 1,...,p,
    # where u_i is the ith row of Uhat.
    # a naive serial implementation is provided as a default, but if a user has implemented the ability to perform
    # these evaluations in parallel, they should supply that function.
    # the return arguments of funxUhat should be:
    # a length p numpy array with ith entry f(x+u_i), and
    # a p times n numpy array with the ith row given as nabla f(x+u_i).

    # Uhat0: An initial finite subset of U. Should be structured as a p times n array where the ith row is u_i\in U.
    # By default, we will use a coordinate-aligned stencil that evaluates at the boundary of x + U.

    # nfmax: a budget on the number of funx evaluations we are willing to perform.

    # sanity checks on input data:
    prob_dim = len(x0)
    if len(b_low) != prob_dim:
        raise ValueError("lower bounds are not the same length as x0")
    if len(b_upp) != prob_dim:
        raise ValueError("upper bounds are not the same length as x0")
    if np.any(b_low > b_upp):
        raise ValueError("lower bounds must be <= upper bounds in all coordinates")
    if Uhat0 is not None:
        for count, u in enumerate(Uhat0):
            if np.any(u < b_low) or np.any(u > b_upp):
                raise ValueError("supplied Uhat0 is not properly contained in U")

    # Step 0: Initialize Uhat and funxUhat if not done already:
    if Uhat0 is None:
        Uhat = np.zeros((2 * prob_dim + 1, prob_dim))
        for coord in range(prob_dim):
            Uhat[coord + 1, coord] = b_low[coord]
            Uhat[prob_dim + coord + 1, coord] = b_upp[coord]
    else:
        Uhat = Uhat0

    if funxUhat is None:
        funxUhat = lambda y, Uhatt: serial_funxUhat(y, Uhatt, funx)

    # Initialize the nlopt wrapper (only necessary if you don't have a better in-place implementation of gradients)
    wrapped_function = lambda y, gradd: nlopt_wrapper(y, gradd, funx)

    # Tell manifold sampling this is a piecewise maximum problem
    hfun = pw_maximum

    # These are for the manifold sampling subproblems: we can generalize this to actually bounded problems eventually
    # if we're careful ...
    LB = -np.inf * np.ones((1, prob_dim))
    UB = np.inf * np.ones((1, prob_dim))

    # while not converged ...
    nf = 0
    x = x0
    gtol_msp = 0.1 # should be largeish, this will shrink over the run of the algorithm
    gtol = 1e-8 # should be a "standard" gradient tolerance for ROBOBOA
    xtol = 1e-12 # says two points are too close together to be considered different in Uhat
    printf = False
    stopped = False
    while nf < nfmax and not stopped:

        #################################### STEP 1 ################################################
        # solve the manifold sampling subproblem defined by Uhat
        Ffun = lambda y: funxUhat(y, Uhat)
        X_msp, F_msp, Grad_msp, h_msp, nf_msp, xkin, flag = manifold_sampling_primal_with_gradients(hfun, Ffun, x, LB, UB, nfmax - nf, "quadprog", gtol_msp, printf)
        x = X_msp[xkin]

        # update the count
        nf = nf + nf_msp * len(Uhat)

        # update gtol_msp before next iteration
        if h_msp[xkin] > h_msp[0] - 0.5 * np.linalg.norm(Grad_msp[xkin]):
            gtol_msp = np.maximum(gtol, 0.5 * gtol_msp)
            if gtol_msp == gtol:
                break
        else:
            gtol_msp = 2.0 * gtol_msp

        if nf >= nfmax:
            break

        #################################### STEP 2 ################################################
        # augment Uhat - what is an approximate maximizer of funx in U?
        # we'll use nlopt.
        # initialize optimizer
        optimizer = nlopt.opt(nlopt.LD_SLSQP, prob_dim)

        # set function
        optimizer.set_max_objective(wrapped_function)

        # set lower and upper bounds
        lower_bounds = x + b_low
        upper_bounds = x + b_upp
        optimizer.set_lower_bounds(lower_bounds)
        optimizer.set_upper_bounds(upper_bounds)

        # TO-DO: set atol or rtol explicitly???
        optimizer.set_xtol_abs(1e-6)
        optimizer.set_xtol_rel(1e-6)

        # optimize
        # randomize the starting point to avoid local maxima:
        y0 = np.zeros(prob_dim)
        for coord in range(prob_dim):
            y0[coord] = np.random.uniform(lower_bounds[coord], upper_bounds[coord], 1)
        xopt = optimizer.optimize(y0)
        robust_value = optimizer.last_optimum_value()

        # add to Uhat
        unew = xopt - x
        if np.amin(np.sum((np.tile(unew, (Uhat.shape[0], 1)) - Uhat)**2, axis=1)) > xtol:
            Uhat = np.vstack((Uhat, xopt - x))
        nf_nlopt = optimizer.get_numevals()
        nf = nf + nf_nlopt
        print("nf: ", nf, "estimate of robust value: ", robust_value, "current gtol: ", gtol_msp)

    return x, Uhat, F_msp[xkin], Grad_msp[xkin], flag




def serial_funxUhat(x, Uhat, funx):
    # Get shape
    Uhat_size, prob_dim = np.shape(Uhat)

    # zero outputs
    Fvec = np.zeros(Uhat_size)
    Jvec = np.zeros((Uhat_size, prob_dim))

    for row, u in enumerate(Uhat):
        fval, gval = funx(x + u)
        Fvec[row] = fval
        Jvec[row] = gval

    return Fvec, Jvec


def nlopt_wrapper(y, gradd, funx):
    f, g = funx(y.copy())
    if gradd.size > 0:
        gradd[:] = g
    return f


