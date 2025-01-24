# This declares the hfun for Test_compare_pounder_pounders_with_jax_CFI.py and
# then used jax to combine the quadratic models of each component of the
# inputs to the hfun.
#
# For other general use cases of pounders on smooth hfuns, only the hfun below
# needs to be changed (and combinemodels_jax can be given to pounders)


import jax
import jax.numpy as jnp
import numpy

jax.config.update("jax_enable_x64", True)


def hfun(z):
    number_of_js = len(z) // 2 # Is also 2^(number_of_qubits)

    d_init = z[:number_of_js]
    d_pert = z[number_of_js:]
    v1 = jnp.sqrt(d_init)
    v2 = jnp.sqrt(d_pert)
    v3 = (v1 - v2)**2
    res = jnp.sum(v3)
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
