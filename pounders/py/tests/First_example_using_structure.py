"""
Unit test of compute function
"""

import ibcdfo.pounders as pdrs
import numpy
import jax
jax.config.update("jax_enable_x64", True)

spsolver = 2
nf_max = 500
g_tol = 1e-5
factor = 10


def hfun(z):
    dim = len(z) // 3
    gamma = z[:dim]
    G_of_gamma_r = z[dim : 2 * dim]
    G_of_gamma_i = z[2 * dim :]
    res = 2 * numpy.sum(gamma * G_of_gamma_r * G_of_gamma_i)
    return res


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


combinemodels = combinemodels_jax


def Ffun(gamma):
    G_of_gamma = numpy.sin(gamma) - numpy.arange(1, len(gamma) + 1) * numpy.cos(gamma) * 1j
    out = numpy.squeeze(numpy.hstack((gamma, numpy.real(G_of_gamma), numpy.imag(G_of_gamma))))
    return out


n = 5
# m = 1
m = 3 * n

X_0 = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5])
Low = -2 * numpy.ones((1, n))  # 1-by-n Vector of lower bounds [zeros(1, n)]
Upp = 2 * numpy.ones((1, n))  # 1-by-n Vector of upper bounds [ones(1, n)]
nfs = 1
F_init = numpy.zeros((1, m))
F_init[0] = Ffun(X_0)
xind = 0
delta = 0.1

Results = {}

Opts = {
    "printf": True,
    "spsolver": spsolver,
    "hfun": hfun,
    "combinemodels": combinemodels,
}
Prior = {"nfs": 1, "F_init": F_init, "X_init": X_0, "xk_in": xind}

[X, F, hF, flag, xk_best] = pdrs.pounders(
    Ffun, X_0, n, nf_max, g_tol, delta, m, Low, Upp, Prior=Prior, Options=Opts, Model={}
)

print(X)

filename = "./first_example" + combinemodels.__name__ + ".npy"

Results = {}
Results["Fvec"] = F
Results["H"] = hF
Results["X"] = X
Results["flag"] = flag
numpy.save(filename, Results)
