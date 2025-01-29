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
from declare_hfun_and_combine_model_with_jax_CFI import hfun, combinemodels_jax
#  from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_uniform_qubit_rotations_Ffun as Ffun
from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_single_qubit_rotations_Ffun as Ffun
from qfi_opt.examples.classical_fisher import state_integrator, distribution
import qfi_opt.spin_models as sm

# simulation parameters
N = 4
model = 'XX'
coupling_exponent = 3
dissipation_rates = 1
layers = 1

# involved in CFI computation:
dphi = 1e-5

# create dictionary from simulation parameters
sim_params = {'N': N, 'model': model, 'coupling_exponent': coupling_exponent, 'dissipation_rates': dissipation_rates, 'dphi': dphi}

# parameter bounds and maximum input params
# note that other Ffun's we end up using have slightly different bounds on the theta parameters at the end of this list
num_thetas = N
bounds = [(0, 1/2), (0, 1/2)] + [(0, 1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(0, np.pi)]

#input_params = np.array(2 * [1/4] + [1/4 if _ % 2 == 0 else 1/2 for _ in range(2 * layers)] + [1/2] + num_thetas * [0])
Low = np.array([entry[0] for entry in bounds])
Upp = np.array([entry[1] for entry in bounds])
input_params = Low + np.random.rand(len(bounds)) * (Upp - Low)

nf_max = 500
g_tol = 1e-4
n = len(input_params)
X_0 = input_params

# I'm too dumb to predetermine m (Juan can help), so I'll just compute an unperturbed distribution at the initial point for now:
simulation_obj = getattr(sm, f'simulate_{model}_chain')
fixed_x = input_params[:n-num_thetas]
rho = simulation_obj(params=fixed_x, num_qubits=N, dissipation_rates=dissipation_rates,
                         coupling_exponent=coupling_exponent)
Sx, _, _ = sm.collective_spin_ops(num_qubits=N)
# when theta = 0, Svarphi is just Sx.
rho_varphi = state_integrator(rho, Sx, np.pi / 2)
unpert_dist = distribution(rho_varphi, N)

# wrapped function so sim_params (and x!) are fixed for a single optimization instance.
# This is going to recompute rho(x) every time it is called, which is super wasteful.
# This is very stupid, but at the scale of the test problem, it's OK and not a huge waste of energy.

def wrapped_Ffun_fixed_x(theta):
    return Ffun(np.concatenate((fixed_x, theta)), sim_params)


# the starting point for this reduced Ffun is just:
X_0 = np.atleast_2d(input_params[n-num_thetas:])
# actually, for this experiment, let's NOT use a random starting theta. Let's start theta in the middle of its domain:
X_0 = (np.pi / 2.0) * np.ones(num_thetas)
# and then let's deliberately make the initial TR cover half of the domain.
delta = np.pi / 4.0

# Likewise the bounds are:
Low = np.atleast_2d(Low[n-num_thetas:])
Upp = np.atleast_2d(Upp[n-num_thetas:])

# now the dimension is just the num_thetas in this reduced problem:
n = num_thetas



hF = {}
for call in range(2):
    if call == 0:
        # Call pounders with m=1 building models of hfun(Ffun(x)) directly (not using structure)
        Ffun_to_use = lambda theta: hfun(wrapped_Ffun_fixed_x(theta))
        m = 1  # not using structure
        Opts = {
            "hfun": lambda F: np.squeeze(F),  # not using structure
            "combinemodels": pdrs.identity_combine,  # not using structure
            "printf": 1  # for debugging
        }
    elif call == 1:
        # Calls pounders to combine models of Ffun components using the derivatives of hfun (obtained by jax)
        Ffun_to_use = lambda theta: wrapped_Ffun_fixed_x(theta)
        m = 2 * len(unpert_dist)
        Opts = {
            "hfun": hfun,  # using structure
            "combinemodels": combinemodels_jax,  # using structure
            "printf": 1  # for debugging.
        }

    [X, F, hF[call], flag, xkin] = pdrs.pounders(Ffun_to_use, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
    # print(X,xkin,hF[call])
    #assert flag == 0, "Didn't reach critical point"


print(f"Using structure uses {len(hF[1])} evals. Not using structure uses {len(hF[0])}")