import jax.numpy as jnp
from jax import jit, grad
from DensityMatrixManifold import DensityMatrixManifold
import ipdb

@jit
def retraction(rho, tol=1e-12):
    # First-order retraction: rho + X, projected back to trace 1 and PSD
    # Step 1: Take a step in ambient space
    Y = rho #+ X
    # ensure that you actually passed rho a trace one matrix (it might not be psd)

    Y = (Y + Y.conj().T) / 2  # enforce Hermitian

    # Step 2: Project to PSD cone via eigendecomposition
    evals, evecs = jnp.linalg.eigh(Y)
    evals_clipped = jnp.clip(evals, 0.0, None)
    Y_proj = evecs @ jnp.diag(evals_clipped) @ evecs.conj().T

    # Step 3: Normalize to trace one
    trace_Y = jnp.trace(Y_proj)
    #if trace_Y <= tol:
    #    raise ValueError("Retracted matrix is numerically zero.")
    rho_new = Y_proj / trace_Y
    rho_new = (rho_new + rho_new.conj().T) / 2  # final symmetrize

    return rho_new

@jit
def qfi(A, G, num_qubits, eps=1e-12):

    # put back into density matrix form:
    rho = A @ A.conj().T
    rho /= jnp.trace(rho)

    # Eigendecomposition of rho
    lambdas, psi = jnp.linalg.eigh(rho)  # lambdas: (n,), psi: (n,n), columns are eigenvectors

    # Compute the matrix elements ⟨ψ_i|G|ψ_j⟩ = psi[:,i]† G psi[:,j]
    G_mat = psi.conj().T @ G @ psi  # This is G in the eigenbasis of rho
    G_abs_squared = jnp.abs(G_mat) ** 2  # |⟨ψ_i|G|ψ_j⟩|²

    # Build the denominator λ_i + λ_j and numerator (λ_i - λ_j)^2
    lambda_i = lambdas[:, None]
    lambda_j = lambdas[None, :]

    denom = lambda_i + lambda_j
    numer = (lambda_i - lambda_j) ** 2

    # Avoid division by zero: mask out zero denominators
    mask = denom > eps
    qfi_matrix = jnp.where(mask, 2 * numer / denom * G_abs_squared, 0.0)

    # Sum over all (i,j) to get the QFI
    return jnp.sum(qfi_matrix) / num_qubits**2


if __name__ == "__main__":
    import qfi_opt.spin_models as sm

    layers = 1
    dim = 3 + 2 * layers
    num_qubits = 5
    model = 'ising'
    coupling_exponent = 6.0
    dissipation_rate = 0.01
    G = sm.collective_op(sm.PAULI_Z, num_qubits) / 2

    def sim_func(params, num_qubits, model, coupling_exponent, dissipation_rate):
        sim_obj = getattr(sm, f'simulate_{model}_chain')
        rho = sim_obj(params=params, num_qubits=num_qubits, dissipation_rates=dissipation_rate,
                      coupling_exponent=coupling_exponent)

        return rho

    sim_func_to_run = lambda params: sim_func(params, num_qubits, model, coupling_exponent, dissipation_rate)

    # pick a parameter point and compute its rho
    params = jnp.ones(dim)
    rho = sim_func_to_run(params)

    # generate a small tangent
    # instantiate a manifold
    manifold = DensityMatrixManifold(jnp.shape(rho)[0])
    # X should be a a trace zero matrix:
    X = manifold.random_tangent_vector(rho)

    #hfun = lambda X: qfi(retraction(rho + X), G, num_qubits)
    # what if no retraction?
    hfun = lambda X: qfi(rho + X, G, num_qubits)
    grad_hfun = jit(grad(hfun, argnums=0))

    obj_val = hfun(X)
    obj_jac = grad_hfun(X)
    print(f"Value: {obj_val}")
    print(f"Gradient: {obj_jac}")