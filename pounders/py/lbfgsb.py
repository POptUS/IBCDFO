from scipy.optimize import minimize
import numpy as np
import ipdb
from scipy.linalg import block_diag
from bayes_opt import BayesianOptimization

def unroll_z_into_matrix_sm(z, sim_params):

    num_qubits = sim_params['num_qubits']

    # reconstruct rho
    rho_shape = 2**num_qubits
    rho = np.zeros((rho_shape, rho_shape), dtype='complex128')

    # populate the real entries of rho:
    Fdim = len(z)
    fdim_idx = 0
    fdim_idx_end = int(Fdim / 2)
    upper_triangular_array = z[fdim_idx:fdim_idx_end]
    upper_triangular_indices = np.triu_indices(rho_shape)
    rho[upper_triangular_indices] = upper_triangular_array

    # populate the complex entries:
    fdim_idx = fdim_idx_end
    fdim_idx_end = Fdim
    upper_triangular_array = z[fdim_idx:fdim_idx_end]
    rho[upper_triangular_indices] += 1j * upper_triangular_array

    # make Hermitian
    rho += rho.T.conj()
    rho[np.diag_indices_from(rho)] /= 2.0

    return rho

def unroll_z_into_matrix(z, sim_params):
    n = sim_params['n']

    # reconstruct rho
    shape_vector = sim_params['shape_vector']
    Jmax = len(shape_vector)

    fdim_idx = 0
    rho = np.empty((Jmax,), dtype=object)
    for jm in range(Jmax):
        subdim = shape_vector[jm]
        rho_block = np.zeros((subdim, subdim), dtype='cdouble')

        # populate the real entries of the matrix:
        fdim_idx_end = int(fdim_idx + subdim * (subdim + 1) / 2)
        upper_triangular_array = z[fdim_idx:fdim_idx_end]
        upper_triangular_indices = np.triu_indices(subdim)
        rho_block[upper_triangular_indices] = upper_triangular_array

        # populate the complex entries of the matrix:
        fdim_idx = fdim_idx_end
        fdim_idx_end = int(fdim_idx + subdim * (subdim + 1) / 2)
        upper_triangular_array = z[fdim_idx:fdim_idx_end]
        rho_block[upper_triangular_indices] += 1j * upper_triangular_array

        # make rho_block Hermitian
        rho_block += rho_block.T.conj()
        rho_block[np.diag_indices_from(rho_block)] /= 2.0

        # append
        rho[jm] = rho_block

        # get ready for next iteration
        fdim_idx = fdim_idx_end

    block_matrix = block_diag(rho[0], rho[1])
    for j in range(2, Jmax):
        block_matrix = block_diag(block_matrix, rho[j])

    return block_matrix

def objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H, sim_params, compute_grad=False):

    n, m = np.shape(G)
    My = np.zeros(m)
    Jy = np.zeros((n, m))

    yG = y @ G

    for i in range(m):  # this can certainly be vectorized, I just want it readable for debugging.
        My[i] = Fx[i] + yG[i] + 0.5 * y @ H[:, :, i] @ y.T
        Jy[:, i] = G[:, i] + H[:, :, i] @ y.T

    if compute_grad:
        hfundMy = hfun_d(My)

        grad = np.zeros(n)
        for j in range(n):
            jth_partial = unroll_z_into_matrix_sm(Jy[j, :], sim_params)
            grad[j] = np.real(np.trace(hfundMy.T @ jth_partial))
        return grad
    else:
        hfunMy = hfun(My)
        return hfunMy


def run_lbfgsb(hfun, hfun_d, Fx, G, H, L, U, sim_params):

    #  create wrapper functions (sooooo stupid, but i want to use scipy for now because i trust LBFGS-B)
    def obj(y):
        hFy = objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H, sim_params, compute_grad=False)
        return hFy

    def jac(y):
        gradhFy = objective_for_lbfgsb(y, hfun, hfun_d, Fx, G, H, sim_params, compute_grad=True)
        return gradhFy

    n, m = np.shape(G)

    x0 = np.zeros(n)
    #x0 = np.random.uniform(L, U)

    hFx0 = obj(x0)

    bounds = [(L[i], U[i]) for i in range(n)]
    options = {"gtol": 1e-12, "ftol": 1e-12}
    out = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options=options, jac=jac)
    Xsp = out.x
    success = out.success
    fval = obj(Xsp)
    if np.linalg.norm(Xsp) > 0 and fval < hFx0:
        mdec = fval - hFx0
        return Xsp, mdec, success
    else:
        # try running LBFGS without the explicit gradient.
        print("Need to do an LBFGS run without gradients because of gradient failure!")
        out = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options=options)
        success = out.success
        Xsp = out.x
        fval = obj(Xsp)
        if np.linalg.norm(Xsp) > 0 and fval < hFx0:
            mdec = out.fun - hFx0
        else:  # solver blew it, try backtracking
            print("Trying a backtrack now because LBFGS without gradients also blew it!")
            g = jac(x0)
            beta = 10
            mdec = 0  # gotta error if the loop below fails
            for j in range(9):
                trial = x0 - (beta ** (-j)) * g
                hftrial = obj(trial)
                if hftrial < hFx0:
                    success = out.success
                    Xsp = trial - x0
                    mdec = hftrial - hFx0
                    break
        return Xsp, mdec, success


def run_bayes_opt(hfun, hfun_d, Fx, G, H, L, U, sim_params, random_seed=888):

    n, m = np.shape(G)
    x0 = np.zeros(n)

    def bayes_wrapped(*args, **kwargs):
        n = len(kwargs)
        x = np.zeros(n)
        for t in range(n):
            x[t] = kwargs['x[' + str(t) + ']']
        return -1.0 * objective_for_lbfgsb(x, hfun, hfun_d, Fx, G, H, sim_params, compute_grad=False)

    hFx0 = -1.0 * objective_for_lbfgsb(x0, hfun, hfun_d, Fx, G, H, sim_params, compute_grad=False)

    bounds = [(L[i], U[i]) for i in range(n)]

    # bayes_opt solver requires naming your variables like this:
    # (i personally believe they do this because it forces you to consider, explicitly, how large your problem is and
    # whether bayesian optimization is appropriate)

    pbounds = {}
    for t in range(n):
        pbounds['x[' + str(t) + ']'] = bounds[t]

    optimizer = BayesianOptimization(
        f=bayes_wrapped,
        pbounds=pbounds,
        random_state=random_seed,
        verbose=1
    )

    optimizer.maximize(
        init_points=n, # intuition: Latin hypercube sampling
        n_iter=10*n
    )

    qfi_value = -1.0 * optimizer.max['target']
    mdec = qfi_value - hFx0
    Xsp = np.zeros(n)
    for t in range(n):
        Xsp[t] = optimizer.max['params']['x[' + str(t) + ']']

    return Xsp, mdec, True
