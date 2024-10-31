"""
This tests pounder (no structure) against pounders with a novel hfun from
quantum. The objective is a function of a real vector gamma, the get a complex
vector output G_of_gamma, and the objective is
imag(sum_{i=1}^m gamma_i*G_of_gamma**2), which is
sum_{i=1}^m imag(gamma_i*G_of_gamma**2), which is
sum_{i=1}^m gamma_i*imag(G_of_gamma**2), which is
sum_{i=1}^m 2 * gamma_i*G_of_gamma_r*G_of_gamma_i 

So given gamma, we compute G_of_gamma and return its real and imaginary parts. 

"""

import ibcdfo.pounders as pdrs
import numpy

import jax

jax.config.update("jax_enable_x64", True)

nf_max = 500
g_tol = 1e-5
n = 5
X_0 = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5])
Low = -2 * numpy.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
Upp = 2 * numpy.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
nfs = 1
delta = 0.1

# Both approaches use the same hfun


def hfun(z):
    dim = len(z) // 3
    gamma = z[:dim]
    G_of_gamma_r = z[dim : 2 * dim]
    G_of_gamma_i = z[2 * dim :]
    res = 2 * numpy.sum(gamma * G_of_gamma_r * G_of_gamma_i)
    return res


# First, we call pounders with m=1, not using structure

m = 1  # not using structure


def Ffun_scalar_out(gamma):
    G_of_gamma = (
        numpy.sin(gamma) - numpy.arange(1, len(gamma) + 1) * numpy.cos(gamma) * 1j
    )
    out = numpy.squeeze(
        numpy.hstack((gamma, numpy.real(G_of_gamma), numpy.imag(G_of_gamma)))
    )
    return hfun(out)


Opts = {
    "hfun": lambda F: numpy.squeeze(F),  # not using structure
    "combinemodels": pdrs.identity_combine,  # not using structure
}

[X, F, hF_without_struct, flag, xk_best] = pdrs.pounders(
    Ffun_scalar_out, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts
)
assert flag == 0, "Didn't reach critical point"


# Then, we use jax to get models of the hFun and we call pounders using structure
m = 3 * n  # using structure


def Ffun_vec_out(gamma):
    G_of_gamma = (
        numpy.sin(gamma) - numpy.arange(1, len(gamma) + 1) * numpy.cos(gamma) * 1j
    )
    out = numpy.squeeze(
        numpy.hstack((gamma, numpy.real(G_of_gamma), numpy.imag(G_of_gamma)))
    )
    return out


@jax.jit
def hfun_d(z, zd):
    resd = jax.jvp(hfun, (z,), (zd,))
    return resd


@jax.jit
def hfun_dd(z, zd, zdt, zdd):
    _, resdd = jax.jvp(hfun_d, (z, zd), (zdt, zdd))
    return resdd


def G_combine(Cres, Gres):
    n, m = Gres.shape
    G = numpy.zeros(n)
    for i in range(n):
        _, G[i] = hfun_d(Cres, Gres[i, :])
    return G


def H_combine(Cres, Gres, Hres):
    n, _, m = Hres.shape
    H = numpy.zeros((n, n))
    for i in range(n):
        for j in range(n):
            _, H[i, j] = hfun_dd(Cres, Gres[i, :], Gres[j, :], Hres[i, j, :])
    return H


def combinemodels_jax(Cres, Gres, Hres):
    return G_combine(Cres, Gres), H_combine(Cres, Gres, Hres)


Opts = {
    "hfun": hfun,  # using structure
    "combinemodels": combinemodels_jax,  # using structure
}

[X, F, hF_with_struct, flag, xk_best] = pdrs.pounders(
    Ffun_vec_out, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Options=Opts
)
assert flag == 0, "Didn't reach critical point"

print(
    f"Using structure uses {len(hF_with_struct)} evals. Not using structure uses {len(hF_without_struct)}"
)
