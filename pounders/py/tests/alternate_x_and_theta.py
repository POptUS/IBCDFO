"""
This tests pounder (no structure) against pounders with a novel hfun arising in a
quantum sensing application. The objective is a function of a two sets of
inputs: d^{init} and d^{pert}

Given these, the objective is
sum_{j=1}^m (sqrt(d^{init}_j) - sqrt(d^{init}_j))**2

This is the hfun. So given x, the Ffun must compute/return d^{init} and d^{pert}
"""

import ibcdfo.pounders as pdrs
import numpy as np
from declare_hfun_and_combine_model_with_jax_CFI import hfun, combinemodels_jax, hfun_d
#from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_uniform_qubit_rotations_Ffun as Ffun
#from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_single_qubit_rotations_Ffun as Ffun
from qfi_opt.examples.classical_fisher import compute_bitstring_basis_CFI_for_uniform_qubit_rotations_Ffun as Ffun
#from qfi_opt.examples.classical_fisher import compute_bitstring_basis_CFI_for_single_qubit_rotations_Ffun as Ffun
#import qfi_opt.spin_models as sm
from bayes_opt import BayesianOptimization
import ipdb

##  Define the problem.
#  simulation parameters
N = 4
model = 'XX'
coupling_exponent = 0.0
dissipation_rates = 1.0
layers = 1

# involved in CFI computation:
dphi = 1e-5

# create dictionary from simulation parameters
sim_params = {'N': N, 'model': model, 'coupling_exponent': coupling_exponent, 'dissipation_rates': dissipation_rates, 'dphi': dphi}

##  Parameter bounds and input params
#  note that other Ffun's we end up using have slightly different bounds on the theta parameters at the end of this list
num_thetas = 1 # should be N for single_qubit_rotations
bounds = [(0, 1/2), (0, 1/2)] + [(0, 1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(0, np.pi)]

Low = np.array([entry[0] for entry in bounds])
Upp = np.array([entry[1] for entry in bounds])
input_params = Low + np.random.rand(len(bounds)) * (Upp - Low)
n = len(input_params)
pbounds = {}
for t in range(num_thetas):
    pbounds['theta['+str(t)+']'] = bounds[n - num_thetas + t]
pbounds_x = {}
for t in range(n - num_thetas):
    pbounds_x['x['+str(t)+']'] = bounds[t]
pbounds_all = {}
for t in range(n - num_thetas):
    pbounds_all['x['+str(t)+']'] = bounds[t]
for t in range(n - num_thetas, n):
    pbounds_all['theta['+str(t - n + num_thetas)+']'] = bounds[t]

bounds = [(-np.inf, np.inf) for _ in range(2 * layers + 3 + num_thetas)]
Low = np.array([entry[0] for entry in bounds])
Upp = np.array([entry[1] for entry in bounds])
#input_params = np.array(2 * [1/4] + [1/4 if _ % 2 == 0 else 1/2 for _ in range(2 * layers)] + [1/2] + num_thetas * [0])

nf_max = 500
g_tol = 1e-4
X_0 = input_params

# I'm too dumb to predetermine m (Juan can help), so I'll just compute an unperturbed distribution at the initial point for now:
#simulation_obj = getattr(sm, f'simulate_{model}_chain')
fixed_x = input_params[:n-num_thetas]
fixed_theta = input_params[n-num_thetas:]


theta_Low = np.atleast_2d(Low[n-num_thetas:])
theta_Upp = np.atleast_2d(Upp[n-num_thetas:])
x_Low = np.atleast_2d(Low[:n-num_thetas])
x_Upp = np.atleast_2d(Upp[:n-num_thetas])
n_theta = num_thetas
n_x = n - num_thetas

rho = Ffun(np.concatenate((fixed_x, fixed_theta)), sim_params, just_return_rho=True)
m = len(Ffun(np.concatenate((fixed_x, fixed_theta)), sim_params, just_theta=True, rho=rho))

Opts = {
    "hfun": hfun,  # using structure
    "combinemodels": combinemodels_jax,
    "hfun_d": hfun_d,  # using structure
    "printf": 1,  # for debugging.
    "spsolver": 4,
    "delta_min": g_tol
}

X_0 = np.atleast_2d(input_params)

# parameters related to x and theta alternating loop
num_iters = 10
stop_tolerance = 1e-5
optimize_over_x_and_theta = True

# trying something
def bayes_wrapped_hFfun(*args, **kwargs):
    n = len(kwargs)
    x = np.zeros(n)
    for t in range(n - num_thetas):
        x[t] = kwargs['x[' + str(t) + ']']
    for t in range(n - num_thetas, n):
        x[t] = kwargs['theta[' + str(t - n + num_thetas) + ']']
    Fvec = Ffun(x, sim_params)
    return -1.0 * hfun(Fvec)

optimizer = BayesianOptimization(
        f=bayes_wrapped_hFfun,
        pbounds=pbounds_all,
        random_state=8,
        verbose=1
    )

## UNCOMMENT WHEN YOU WANT TO START WITH AN APPROXIMATE GLOBAL OPT SOLUTION
#print("Starting a Bayesian optimization run to maximize over theta.")
#optimizer.maximize(
#    init_points=n,
#    n_iter=np.maximum(2**n, 16) # intuition: let an acquisition function at least explore the corners.
#)

#f_updated = -1.0 * optimizer.max['target']

#for t in range(n - num_thetas):
#    input_params[t] = optimizer.max['params']['x[' + str(t) + ']']
#for t in range(n-num_thetas, n):
#    input_params[t] = optimizer.max['params']['theta[' + str(n - num_thetas + t) + ']']


for iter in range(num_iters):

    # First optimize over theta, fixing rho
    # wrapped function so sim_params (and x!) are fixed for a single optimization instance
    def wrapped_Ffun_fixed_x(theta):
        return Ffun(np.concatenate((fixed_x, theta)), sim_params, just_theta=True, rho=rho)

    def bayes_wrapped_hFfun_fixed_x(*args, **kwargs):
        num_thetas = len(kwargs)
        theta = np.zeros(num_thetas)
        for t in range(num_thetas):
            theta[t] = kwargs['theta['+str(t)+']']
        Fvec = Ffun(np.concatenate((fixed_x, theta)), sim_params, just_theta=True, rho=rho)
        return -1.0 * hfun(Fvec)

    # the starting point for this reduced Ffun is just:
    theta_0 = input_params[n-num_thetas:]

    # the starting value:
    if iter == 0:
        f_incumbent = hfun(wrapped_Ffun_fixed_x(theta_0))

    # we reinstantiate this object because the objective changes in each pass through the loop (fixed_x changes)
    optimizer = BayesianOptimization(
        f=bayes_wrapped_hFfun_fixed_x,
        pbounds=pbounds,
        random_state=8,
        verbose=1
    )

    print("Starting a Bayesian optimization run to maximize over theta.")
    optimizer.maximize(
        init_points=num_thetas,
        n_iter=np.maximum(2**num_thetas, 16) # intuition: let an acquisition function at least explore the corners.
    )

    f_updated = -1.0 * optimizer.max['target']

    if f_updated < f_incumbent:
        print("Bayesian optimization run completed, best function value is now ", f_updated)
        for t in range(num_thetas):
            input_params[n - num_thetas + t] = optimizer.max['params']['theta[' + str(t) +']']
        fixed_theta = input_params[n - num_thetas:]
        # reset delta, we had a jump
        delta_theta = np.pi / 4.0
    else:
        print("Bayesian optimization run completed, function value not improved.")

    theta_0 = input_params[n-num_thetas:]

    # Now we clean up that Bayesian optimization's approximation of a global max with a pounders run:
    if num_thetas > 1:
        # let's deliberately make the initial TR cover half of the domain.
        if iter == 0:
            delta_theta = np.pi / 4.0
        print("Starting a POUNDers run to refine approximate theta maximizer.")
        [X, F, hF, flag, xkin] = pdrs.pounders(wrapped_Ffun_fixed_x, theta_0, n_theta, nf_max, g_tol, delta_theta, m, theta_Low, theta_Upp, Options=Opts)
        # default delta for next run unless global optimization moves us away:
        delta = 10 * g_tol
        input_params[n - num_thetas:] = X[xkin]
        fixed_theta = X[xkin]
        f_incumbent = hF[xkin]
        print("POUNDers run completed, best function value is now ", f_incumbent)

    if optimize_over_x_and_theta:
        # Now optimize locally over both x and theta
        # wrapped function so sim_params (and theta!) are fixed for a single optimization instance
        def wrapped_Ffun(x):
            return Ffun(x, sim_params)

        # the starting point for this reduced Ffun is just:
        x_0 = input_params

        if iter == 0:
            # let's deliberately make the initial TR cover half of the domain.
            delta_x = 0.125

        print("Starting a POUNDers run to minimize over x and theta simultaneously.")
        [X, F, hF, flag, xkin] = pdrs.pounders(wrapped_Ffun, x_0, n, nf_max, g_tol, delta_x, m, Low,
                                               Upp, Options=Opts)
        input_params = X[xkin]
        fixed_x = input_params[:n-num_thetas]
        fixed_theta = input_params[n-num_thetas:]
        ################################################################################################
    else:
        # Now optimize over x, fixing theta:
        # wrapped function so sim_params (and theta!) are fixed for a single optimization instance
        def wrapped_Ffun_fixed_theta(x):
            return Ffun(np.concatenate((x, fixed_theta)), sim_params)

        # the starting point for this reduced Ffun is just:
        x_0 = input_params[:n - num_thetas]
        if iter == 0:
            # let's deliberately make the initial TR cover half of the domain.
            delta_x = 0.125

        print("Starting a POUNDers run to minimize over x with fixed theta.")
        [X, F, hF, flag, xkin] = pdrs.pounders(wrapped_Ffun_fixed_theta, x_0, n_x, nf_max, g_tol, delta_x, m, x_Low,
                                            x_Upp, Options=Opts)
    f_incumbent = hF[xkin]
    # default delta:
    delta_x = 10 * g_tol
    print("POUNDers run completed, best function value is now ", f_incumbent)

    # check stopping conditions:
    if iter == 0:
        f_best = f_incumbent
        if optimize_over_x_and_theta:
            input_params = X[xkin]
            fixed_x = input_params[:n - num_thetas]
            fixed_theta = input_params[n-num_thetas:]
        else:
            input_params[:n - num_thetas] = X[xkin]
            fixed_x = X[xkin]
    else:
        if f_best - f_incumbent < stop_tolerance:
            if iter < num_iters:
                # try a global optimization run
                def bayes_wrapped_hFfun_fixed_theta(*args, **kwargs):
                    num_xs = len(kwargs)
                    x = np.zeros(num_xs)
                    for t in range(num_xs):
                        x[t] = kwargs['x[' + str(t) + ']']
                    Fvec = Ffun(np.concatenate((x, fixed_theta)), sim_params)
                    return -1.0 * hfun(Fvec)

                # we reinstantiate this object because the objective changes in each pass through the loop (fixed_theta changes)
                optimizer = BayesianOptimization(
                    f=bayes_wrapped_hFfun_fixed_theta,
                    pbounds=pbounds_x,
                    random_state=8,
                    verbose=1
                )

                print("Convergence possible, starting a Bayesian optimization run to maximize over x to attempt to verify global optimality.")
                optimizer.maximize(
                    init_points=n - num_thetas,
                    n_iter=2 ** (n - num_thetas)  # intuition: let an acquisition function at least explore the corners.
                )

                f_updated = -1.0 * optimizer.max['target']
                if f_updated < hF[xkin]: # did we improve over the last pounders run?
                    print("Bayesian optimization completed, better x found, continuing optimization. Best function value is now ", f_updated)
                    for t in range(n - num_thetas):
                        input_params[t] = optimizer.max['params']['x[' + str(t) + ']']
                    fixed_x = input_params[:n - num_thetas]

                    x_0 = input_params[:n - num_thetas]
                    f_incumbent = f_updated

                    # reset delta, we had a jump
                    delta_x = 0.125
                else: # we didn't do better after global opt attempt, just give up
                    print("Bayesian optimization completed, better x not found, stopping with optimal value ", f_incumbent)
                    break
        else:
            f_best = hF[xkin]
            if optimize_over_x_and_theta:
                input_params = X[xkin]
                fixed_x = input_params[:n - num_thetas]
                fixed_theta = input_params[n - num_thetas:]
            else:
                input_params[:n - num_thetas] = X[xkin]
                fixed_x = X[xkin]

    # Recompute rho for next iteration
    rho = Ffun(np.concatenate((fixed_x, fixed_theta)), sim_params, just_return_rho=True)


