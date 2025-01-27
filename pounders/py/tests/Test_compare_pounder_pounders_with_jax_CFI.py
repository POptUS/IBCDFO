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
from qfi_opt.examples.classical_fisher import compute_collective_basis_CFI_for_uniform_qubit_rotations_Ffun as Ffun
from qfi_opt.examples.classical_fisher import state_integrator, distribution
import qfi_opt.spin_models as sm
import ipdb

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


# wrapped function so sim_params are fixed for a single optimization instance.
def wrapped_Ffun(params):
    return Ffun(params, sim_params)


# parameter bounds and maximum input params
# note that other Ffun's we end up using have slightly different bounds on the theta parameters at the end of this list
num_thetas = 1
bounds = [(0, 1/2), (0, 1/2)] + [(0, 1/2) if _ % 2 == 0 else (0, 1) for _ in range(2 * layers)] + [(0, 1)] + num_thetas * [(-np.inf, np.inf)]
input_params = np.array(2 * [1/2] + [1/2 if _ % 2 == 0 else 1 for _ in range(2 * layers)] + [1] + num_thetas * [0])

# def Ffun(x, nostruct=True):
#     # This is a synthetic Ffun. The real example calls an expensive-to-evaluate
#     # quantum system to obtain d^{init} and d^{pert}
#     number_of_js = 2 ** len(x)
#
#     # Define d_init as a sinusoidal function of x, scaled by the index
#     d_init = np.array([1 + np.sin(i + 1 + np.sum(x)) for i in range(number_of_js)])
#
#     # Define d_pert as a polynomial function of x
#     d_pert = np.array([np.sum(x) ** (2) + (i + 1) ** 0.5 for i in range(number_of_js)])
#
#     out = np.squeeze(np.hstack((d_init, d_pert)))
#     if nostruct:
#         return hfun(out)
#     else:
#         return out


nf_max = 500
g_tol = 1e-5
n = len(input_params)
X_0 = input_params
Low = np.array([entry[0] for entry in bounds])
Upp = np.array([entry[1] for entry in bounds])
delta = 0.1

# I'm too dumb to predetermine m (Juan can help), so I'll just compute an unperturbed distribution at the initial point for now:
simulation_obj = getattr(sm, f'simulate_{model}_chain')
rho = simulation_obj(params=input_params[:n-1-num_thetas], num_qubits=N, dissipation_rates=dissipation_rates,
                         coupling_exponent=coupling_exponent)
Sx, _, _ = sm.collective_spin_ops(num_qubits=N)
# when theta = 0, Svarphi is just Sx.
rho_varphi = state_integrator(rho, Sx, np.pi / 2)
unpert_dist = distribution(rho_varphi, N)
m = 2 * len(unpert_dist)

hF = {}
for call in [1]: #range(2):
    if call == 0:
        # Call pounders with m=1 building models of hfun(Ffun(x)) directly (not using structure)
        Ffun_to_use = lambda x: wrapped_Ffun(x)
        m = 1  # not using structure
        Opts = {
            "hfun": lambda F: np.squeeze(F),  # not using structure
            "combinemodels": pdrs.identity_combine,  # not using structure
        }
    elif call == 1:
        # Calls pounders to combine models of Ffun components using the derivatives of hfun (obtained by jax)
        Ffun_to_use = lambda x: wrapped_Ffun(x)
        Opts = {
            "hfun": hfun,  # using structure
            "combinemodels": combinemodels_jax,  # using structure
            "printf": 1  # for debugging.
        }

    [X, F, hF[call], flag, xkin] = pdrs.pounders(Ffun_to_use, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts)
    # print(X,xkin,hF[call])
    assert flag == 0, "Didn't reach critical point"


print(f"Using structure uses {len(hF[1])} evals. Not using structure uses {len(hF[0])}")
assert len(hF[1]) < len(hF[0]), "While not true for every problem, using structure on this problem should be beneficial"
