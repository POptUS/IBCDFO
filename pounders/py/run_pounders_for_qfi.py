import ibcdfo.pounders as pdrs
import numpy as np
import ipdb
from density_Ffun import rollup
from lbfgsb import unroll_z_into_matrix_sm

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
    #Model = {"np_max": n + 1, "Par": Pars}
    #Model = {"Par": Pars}
    Model = {}

    # don't actually bound the pounders run (function is periodic in all variables)
    bounds = [(0, 1 / 2) for _ in range(2)] + [(0, 1 / 2) if _ % 2 == 0 else (0, 1) for _ in
                                               range(2 * sim_params['layers'])] + [(0, 1)]
    Low = np.array([entry[0] for entry in bounds])
    Upp = np.array([entry[1] for entry in bounds])

    [X, F, hF, flag, xkin] = pdrs.pounders(Ffun, initial_point, n, nf_max, g_tol, delta_0, m, Low, Upp,
                                           Options=Opts, Model=Model, Prior=Prior)

    return X, F, hF, flag, xkin


if __name__ == "__main__":
    import qfi_opt.spin_models as sm
    from density_Ffun import sm_density_Ffun as density_Ffun
    from retraction_qfi_hfun import qfi, retraction
    from jax import jit, grad

    ##  Define the problem.
    layers = 5
    n = 3 + 2 * layers
    num_qubits = 2
    model = 'ising'
    coupling_exponent = 3.0
    dissipation_rate = 0.01
    G = sm.collective_op(sm.PAULI_Z, num_qubits) / 2

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
    hfun = lambda X: -1.0 * qfi(retraction(X), G, num_qubits)
    hfun_d = jit(grad(hfun, argnums=0))
    rollup_hfun = lambda F: hfun(unroll_z_into_matrix_sm(F, sim_params))
    rollup_hfun_d = lambda F: hfun_d(unroll_z_into_matrix_sm(F, sim_params))

    # three more parameters for pounders:
    delta_0 = 1e-2
    delta_max = []
    Prior = None

    nf_max = 50 * n  # this provides a lower bound on function evaluations per optimizer call

    ## call pounder
    #print("Running pounder (no s!) to find a stationary point of the composite objective function.")

    #hF = lambda x: rollup_hfun(density_Ffun(x, sim_params))
    #trivial_hfun = lambda F: np.squeeze(F)
    #spsolver = 2
    #m = 1
    #X, F, hF, flag, xkin = run_pounders(initial_point, hF, trivial_hfun, [], sim_params, m, delta_0, spsolver,
    #                                    Prior=Prior, delta_max=delta_max, nf_max=nf_max, g_tol=1e-4)
    #x_opt = X[xkin]
    #qfi_after_pounders = -1.0 * hF[xkin]
    #print("Estimate of optimal QFI: ", qfi_after_pounders)

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
