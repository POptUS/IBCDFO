import sys
import pickle
import ibcdfo.pounders as pdrs
import numpy as np
import ipdb
from lbfgsb import unroll_z_into_matrix_sm, unroll_z_into_matrix
import qfi_opt.spin_models as sm
from jax import jit, grad
import qfi_opt.examples.PermSolver_methods as methods


def run_pounders(initial_point, Ffun, hfun, hfun_d, sim_params, m, delta_0=0.125, spsolver=4, Prior=None, nf_max=500, g_tol=1e-4, delta_max=[]):

    n = len(initial_point)

    if spsolver == 4:
        combinemodels = []
    else:
        combinemodels = pdrs.identity_combine

    Opts = {
        "hfun": hfun,  # using structure
        "combinemodels": combinemodels, # not actually used, make sure this doesn't cause errors downstream
        "hfun_d": hfun_d,  # using structure
        "printf": 1,  # for debugging.
        "spsolver": spsolver,
        "delta_min": 1e-6
    }

    if delta_max:
        Opts["delta_max"] = delta_max

    # this is QFI specific:
    Opts["sim_params"] = sim_params

    #Pars = [np.sqrt(n), np.sqrt(n), 1e-5, 0.001] # the second number is forcing us to pick points closer to TR.
    #Model = {"np_max": int((n + 1) * (n + 2) / 2), "Par": Pars}
    #Model = {"np_max": n + 1}
    #Model = {"Par": Pars}
    Model = {}

    bounds = [(0, 1 / 2) for _ in range(2)] + [(0, 1 / 2) if _ % 2 == 0 else (0, 1) for _ in
                                               range(2 * sim_params['layers'])] + [(0, 1)]
    Low = np.array([entry[0] for entry in bounds])
    Upp = np.array([entry[1] for entry in bounds])

    [X, F, hF, flag, xkin] = pdrs.pounders(Ffun, initial_point, n, nf_max, g_tol, delta_0, m, Low, Upp,
                                           Options=Opts, Model=Model, Prior=Prior)

    return X, F, hF, flag, xkin


num_qubits = int(sys.argv[1])
model = sys.argv[2]
dissipation_rate = float(sys.argv[3])
layers = int(sys.argv[4])
coupling_exponent = float(sys.argv[5])
n = 3 + 2 * layers

if coupling_exponent > 0.0:
    G = sm.collective_op(sm.PAULI_Z, num_qubits) / 2
else:
    G = getattr(methods, 'MatSz')(num_qubits // 2)

# create dictionary from simulation parameters
sim_params = {'G': G,
              'num_qubits': num_qubits,
              'dissipation_rate': dissipation_rate,
              'model': model,
              'coupling_exponent': coupling_exponent,
              'layers': layers}

initial_point = np.array(2 * [1 / 4] + [1 / 4 if _ % 2 == 0 else 1 / 2 for _ in range(2 * layers)] + [1 / 2])
#initial_point = np.random.uniform(-np.ones(n), np.ones(n))

# specify hfun and hfun_d
hfun = lambda X: -1.0 * qfi(X, G, sim_params)
hfun_d = jit(grad(hfun, argnums=0))
if coupling_exponent == 0.0:
    rollup_hfun = lambda F: hfun(unroll_z_into_matrix(F, sim_params))
    rollup_hfun_d = lambda F: hfun_d(unroll_z_into_matrix(F, sim_params))

    from density_Ffun import permsolver_density_Ffun as density_Ffun
    from qfi_opt.examples import PermSolver_matrix as matrix
    from retraction_qfi_hfun import qfi_permsolver as qfi

    # map short-range to infinite-range
    parent_models = {"ising": "OAT", "XX": "OAT", "local_TAT": "TAT"}
    # construct Hamiltonian for infinite-range model
    Hamiltonian_set = getattr(matrix, f'{parent_models[model]}Mat')(1.0, num_qubits // 2)
    sim_params['Hmat_set'] = Hamiltonian_set

    # temporary: you have to do one evaluation of this function:
    _, sim_params = density_Ffun(initial_point, sim_params)
else:
    rollup_hfun = lambda F: hfun(unroll_z_into_matrix_sm(F, sim_params))
    rollup_hfun_d = lambda F: hfun_d(unroll_z_into_matrix_sm(F, sim_params))
    from density_Ffun import sm_density_Ffun as density_Ffun
    from retraction_qfi_hfun import qfi

# three more parameters for pounders:
delta_0 = 1e-2
delta_max = []
Prior = None

nf_max = 50 * n  # this provides a lower bound on function evaluations per optimizer call

do_run_pounder = True
do_run_pounders = True

if do_run_pounder:
    # call pounder
    print("Running pounder (no s!) to find a stationary point of the composite objective function.")

    hF = lambda x: rollup_hfun(density_Ffun(x, sim_params))
    trivial_hfun = lambda F: np.squeeze(F)
    spsolver = 2
    m = 1
    X, F, hF, flag, xkin = run_pounders(initial_point, hF, trivial_hfun, [], sim_params, m, delta_0, spsolver,
                                        Prior=Prior, delta_max=delta_max, nf_max=nf_max, g_tol=1e-4)
    x_opt = X[xkin]
    qfi_after_pounders = -1.0 * hF[xkin]
    print("Estimate of optimal QFI: ", qfi_after_pounders)

    data_to_dump = {"X": X, "F": F, "hF": hF, "xkin": xkin, "flag": flag}
    file = open(f"Pickle_files/pounder_{num_qubits}_{model}_{int(100*dissipation_rate)}_{layers}_{int(coupling_exponent)}.pickle", "wb")
    pickle.dump(data_to_dump, file)
    file.close()

if do_run_pounders:
    # call pounders
    print("Running pounders to find a stationary point of the composite objective function.")
    spsolver = 4 #4 for LBFGS
    # a test call, also to get the correct dimension (m) of Fvec
    wrapped_density_Ffun = lambda X: density_Ffun(X, sim_params)
    initial_Fvec = wrapped_density_Ffun(initial_point)
    m = len(initial_Fvec)

    X, F, hF, flag, xkin = run_pounders(initial_point, wrapped_density_Ffun, rollup_hfun, rollup_hfun_d, sim_params, m, delta_0, spsolver,
                                        Prior=Prior, delta_max=delta_max, nf_max=nf_max, g_tol=1e-4)

    x_opt = X[xkin]
    qfi_after_pounders = -1.0 * hF[xkin]
    print("Estimate of optimal QFI: ", qfi_after_pounders)

    data_to_dump = {"X": X, "F": F, "hF": hF, "xkin": xkin, "flag": flag}
    file = open(
        f"Pickle_files/pounders_{num_qubits}_{model}_{int(100 * dissipation_rate)}_{layers}_{int(coupling_exponent)}.pickle",
        "wb")
    pickle.dump(data_to_dump, file)
    file.close()

