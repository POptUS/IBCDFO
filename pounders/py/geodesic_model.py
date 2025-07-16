import sys

from phi2eval import natural_basis
import numpy as np
import ipdb
import jax.numpy as jnp
from jax import jit
import pymanopt
from DensityMatrixManifold import DensityMatrixManifold

def compute_lagrange_polynomials(Y):

    # This function will take an input set of points Y (a [num_points x dim] array) and return the coefficients of
    # Lagrange polynomials in the basis implied by phi2eval

    # get shape of Y
    num_points, dim = np.shape(Y)

    # basis dimension is quadratic in dim
    basis_dim = int(0.5 * (dim + 1) * (dim + 2))

    # initialize Lagrange polynomials via natural basis
    coeffs = np.eye(basis_dim)

    # express Y in natural basis
    Phi = natural_basis(Y)

    # we need to return a permutation for Y
    permutation = np.arange(num_points)

    for idx in range(num_points):
        # evaluate idx'th Lagrange polynomial at every point in Y
        ell_i_at_Y = Phi[idx:] @ coeffs[:, idx]
        largest_ind = np.argmax(np.abs(ell_i_at_Y))
        ell_i_val = ell_i_at_Y[largest_ind]
        if ell_i_val == 0:
            raise RuntimeError("Cannot compute Lagrange polynomials because the set Y is not poised.")
        else:
            # swap the largest_ind and idx rows
            Phi[[idx, idx + largest_ind]] = Phi[[idx + largest_ind, idx]]
            permutation[[idx, idx + largest_ind]] = permutation[[idx + largest_ind, idx]]

        # normalize the idx'th polynomial
        coeffs[:, idx] /= ell_i_val

        # orthogonalize against the newly normalized polynomial:
        lagrange_polys_at_idx = Phi[idx] @ coeffs
        for jdx in range(basis_dim):
            if idx != jdx:
                coeffs[:, jdx] -= lagrange_polys_at_idx[jdx] * coeffs[:, idx]

    return coeffs[:, :num_points].T, permutation


#@jit
def sqrt_psd_jax(A, tol=1e-12):
    """
    Compute matrix square root of a Hermitian PSD matrix using eigendecomposition (JAX).
    """
    A = (A + A.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals_clipped = np.clip(eigvals, a_min=tol, a_max=None)
    sqrt_diag = np.sqrt(eigvals_clipped)
    return eigvecs @ np.diag(sqrt_diag) @ eigvecs.conj().T


#@jit
def bures_distance(rho, sigma, tol=1e-12):
    """
    Compute the Bures distance between two PSD, trace-1 density matrices using JAX.
    Returns a differentiable scalar.

    Args:
        rho: (n, n) Hermitian PSD JAX array
        sigma: (n, n) Hermitian PSD JAX array

    Returns:
        Scalar Bures distance (differentiable)
    """
    rho = (rho + rho.conj().T) / 2
    sigma = (sigma + sigma.conj().T) / 2

    sqrt_rho = sqrt_psd_jax(rho, tol)
    middle = sqrt_rho @ sigma @ sqrt_rho
    middle = (middle + middle.conj().T) / 2

    sqrt_middle = sqrt_psd_jax(middle, tol)

    fidelity = np.real(np.trace(sqrt_middle)) ** 2
    fidelity = np.clip(fidelity, 0.0, 1.0)

    return np.sqrt(2.0 * (1.0 - np.sqrt(fidelity)))


#@jit
def evaluate_lagrange_polys_at_s(s, lagrange_coeffs):
    """
    :param s: a zero-centered step in ansatz parameters
    :param lagrange_coeffs: coefficients defining second-order Lagrange polynomials
    :return: poly_vals: an array of Lagrange polynomial values evaluated at s
    """

    num_points = len(lagrange_coeffs)
    n = len(s)

    #inds_to_use_in_H = jnp.triu_indices(n)
    #other_inds_to_use_in_H = jnp.tril_indices(n)
    inds_to_use_in_H = np.triu_indices(n)

    poly_vals = np.zeros(num_points)
    #poly_vals = jnp.zeros(num_points)

    for poly in range(num_points):
        c = lagrange_coeffs[poly][0]
        g = lagrange_coeffs[poly][1: (n + 1)]
        H = np.zeros((n, n))
        #H = jnp.zeros((n, n))
        #H = H.at[inds_to_use_in_H].set(lagrange_coeffs[poly][(n+1):])
        #H = H.at[other_inds_to_use_in_H].set(lagrange_coeffs[poly][(n+1):])
        #H -= jnp.diag(0.5 * jnp.diag(H))
        #poly_vals = poly_vals.at[poly].set(c + g @ s + 0.5 * s @ H @ s)
        H[inds_to_use_in_H] = lagrange_coeffs[poly][(n + 1):]
        H.T[inds_to_use_in_H] = lagrange_coeffs[poly][(n + 1):]
        poly_vals[poly] = c + g @ s + 0.5 * s @ H @ s

    return poly_vals


#@jit
def geodesic_cost_function(sigma, poly_vals, F):
    """
    The cost function for determining the interpolant at a set of ansatz parameters.
    :param sigma: a density matrix
    :param poly_vals: values of lagrange polynomials at each interpolation point
    :param F: array of interpolated density matrices corresponding to Lagrange polynomials
    :return: cost: cost function value (scalar)
    """

    # the first entry of F must be the interpolant at s = 0:
    num_points = len(F)

    cost = 0.0
    for j in range(num_points):
        rho = F[j]
        cost += poly_vals[j] * bures_distance(rho, sigma) ** 2

    return cost

def compute_model_value(s, lagrange_coeffs, F):
    """
    Computes model value at s by minimizing the geodesic cost function over the manifold of density matrices.
    :param s: a zero-centered step in ansatz parameters
    :param lagrange_coeffs: coefficients defining second-order Lagrange polynomials
    :param F: array of interpolated density matrices corresponding to Lagrange polynomials
    :return: mval: value of interpolation model at s
    """

    # sizes
    dim = np.shape(F[0])[0]

    # get Lagrange polynomial values
    poly_vals = evaluate_lagrange_polys_at_s(s, lagrange_coeffs)

    # instantiate Manifold object for pymanopt
    manifold = DensityMatrixManifold(dim)

    # set up objective function
    #@pymanopt.function.jax(manifold)

    @pymanopt.function.numpy(manifold)
    def objective(density_matrix):
        cost = geodesic_cost_function(density_matrix, poly_vals, F)

        return cost

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(density_matrix):
        # experimenting, just do a finite difference gradient for now
        fd = 1e-8  # finite difference parameter
        n = np.shape(density_matrix)[0]

        cost = geodesic_cost_function(density_matrix, poly_vals, F)
        euclidean_gradient = np.zeros_like(density_matrix)

        # off diagonal perturbations:
        for i in range(n):
            for j in range(i):
                Y = np.copy(density_matrix)
                Y[i, j] += fd * 1j
                costh = geodesic_cost_function(Y, poly_vals, F)
                euclidean_gradient[i, j] += np.imag(costh) / fd

        # symmetrize
        euclidean_gradient += euclidean_gradient.conj().T

        # diagonal entries
        for i in range(n):
            Y = np.copy(density_matrix)
            Y[i, i] += fd * 1j
            costh = geodesic_cost_function(Y, poly_vals, F)
            euclidean_gradient[i, i] += np.imag(costh) / fd

        return euclidean_gradient

    # set up pymanopt problem object
    problem = pymanopt.Problem(manifold, objective, euclidean_gradient=euclidean_gradient)

    # identify pymanopt optimizer - experiment with this!
    optimizer = pymanopt.optimizers.SteepestDescent(min_step_size=1e-16)
    #optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient()
    #optimizer = pymanopt.optimizers.nelder_mead.NelderMead(max_cost_evaluations=10000)

    # run pymanopt
    ipdb.set_trace()
    result = optimizer.run(problem, initial_point=F[2])
    ipdb.set_trace()

    return result.point


def test_example(s, dim, sim_func):

    # generate a (w.h.p. poised for underdetermined interpolation) set of random points in dim-space
    Y = np.concatenate((np.zeros((1, dim)), np.eye(dim)))
    Y = np.concatenate((Y, np.random.randn(dim, dim)))  # 2 * dim + 1 rows in Y

    # compute lagrange polynomials
    lagrange_coeffs, perm = compute_lagrange_polynomials(Y)

    # have to permute back
    Y = Y[perm]

    # generate the dataset F of density matrices using spin models from QFI-Opt:
    F = []
    num_points = len(Y)
    for point in range(num_points):
        # compute here
        F.append(sim_func(Y[point]))

    ## now compute the cost function at each of the F
    ## get Lagrange polynomial values
    #poly_vals = evaluate_lagrange_polys_at_s(s, lagrange_coeffs)
    ## get a random point
    #dim = np.shape(F[0])[0]
    #manifold = DensityMatrixManifold(dim)
    #random_matrix = manifold.random_point()
    #print("cost: ", geodesic_cost_function(random_matrix, poly_vals, F))

    model_value = compute_model_value(s, lagrange_coeffs, F)

    print("model_value: ", model_value)


if __name__ == "__main__":
    import qfi_opt.spin_models as sm
    layers = 1
    dim = 3 + 2 * layers
    s = 0.01 * np.ones(dim)
    num_qubits = 2
    model = 'ising'
    coupling_exponent = 3.0
    dissipation_rate = 0.01

    def sim_func(params, num_qubits, model, coupling_exponent, dissipation_rate):
        sim_obj = getattr(sm, f'simulate_{model}_chain')
        rho = sim_obj(params=params, num_qubits=num_qubits, dissipation_rates=dissipation_rate,
                      coupling_exponent=coupling_exponent)

        return rho

    sim_func_to_run = lambda params: sim_func(params, num_qubits, model, coupling_exponent, dissipation_rate)

    test_example(s, dim, sim_func_to_run)







