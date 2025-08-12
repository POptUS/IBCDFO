import qfi_opt.spin_models as sm
import numpy as np
import qfi_opt.examples.PermSolver_methods as methods
from scipy.linalg import sqrtm

def rollup(rho):

    # determine the correct length (Fdim) of the F vector
    matrix_size = np.shape(rho)[0]
    Fdim = int(matrix_size * (matrix_size + 1))
    Fvec = np.zeros(Fdim)

    upper_triangular_indices = np.triu_indices(matrix_size)
    upper_triangular_array = rho[upper_triangular_indices]

    # store the real part of the array first
    fdim_idx = 0
    fdim_idx_end = int(Fdim / 2)
    Fvec[fdim_idx:fdim_idx_end] = np.real(upper_triangular_array)

    # now the imaginary part:
    fdim_idx = fdim_idx_end
    fdim_idx_end = Fdim
    Fvec[fdim_idx:fdim_idx_end] = np.imag(upper_triangular_array)

    return Fvec


def take_square_root(rho):

    # ensure hermitian:
    rho = 0.5 * (rho + rho.T.conj())

    # compute matrix sqrt
    A = sqrtm(rho)

    return A


def sm_density_Ffun(params, sim_params):

    # unpack necessary attributes from sim_params
    num_qubits = sim_params['num_qubits']
    coupling_exponent = sim_params['coupling_exponent']
    dissipation_rate = sim_params['dissipation_rate']
    model = sim_params['model']

    # compute rho
    sim_obj = getattr(sm, f'simulate_{model}_chain')
    rho = sim_obj(params=params, num_qubits=num_qubits, dissipation_rates=dissipation_rate,
                  coupling_exponent=coupling_exponent)

    A = take_square_root(rho)

    return rollup(A)


def permsolver_density_Ffun(params, sim_params):

    # compute rho
    params = np.squeeze(params.T)

    rho = methods.simulate_layers(params=params,
                                  num_qubits=sim_params['num_qubits'],
                                  Hamiltonian_set=sim_params['Hmat_set'],
                                  dissipation_rates=sim_params['dissipation_rate'])

    return_new_params_flag = False
    if 'Fdim' in sim_params:
        Fdim = sim_params['Fdim']
        shape_vector = sim_params['shape_vector']
        num_blocks = len(shape_vector)
    else:
        # let's populate these missing entries
        Fdim = 0
        num_blocks = len(rho)
        shape_vector = np.zeros(num_blocks, dtype='int')
        for nb in range(num_blocks):
            subdim = np.shape(rho[nb])[0]
            shape_vector[nb] = subdim
            Fdim += subdim * (subdim + 1) # this is the size of the upper triangular part * 2
        sim_params['Fdim'] = Fdim
        sim_params['shape_vector'] = shape_vector
        return_new_params_flag = True

    Fvec = np.zeros(Fdim)
    fdim_idx = 0
    for nb in range(num_blocks):
        subdim = np.shape(rho[nb])[0]
        fdim_idx_end = int(fdim_idx + subdim * (subdim + 1) / 2)

        A = take_square_root(rho[nb])

        upper_triangular_indices = np.triu_indices(rho[nb].shape[0])
        upper_triangular_array = A[upper_triangular_indices]

        # store the real part of the array first
        Fvec[fdim_idx:fdim_idx_end] = np.real(upper_triangular_array)

        fdim_idx = int(fdim_idx + subdim * (subdim + 1) / 2)
        fdim_idx_end = int(fdim_idx_end + subdim * (subdim + 1) / 2)

        # now store the imaginary part of the array
        Fvec[fdim_idx:fdim_idx_end] = np.imag(upper_triangular_array)

        fdim_idx = fdim_idx_end

    if return_new_params_flag:
        return Fvec, sim_params
    else:
        return Fvec
