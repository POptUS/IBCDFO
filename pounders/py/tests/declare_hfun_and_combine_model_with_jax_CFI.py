# This declares the hfun for Test_compare_pounder_pounders_with_jax_CFI.py and
# then used jax to combine the quadratic models of each component of the
# inputs to the hfun.
#
# For other general use cases of pounders on smooth hfuns, only the hfun below
# needs to be changed (and combinemodels_jax can be given to pounders)


import jax
import jax.numpy as jnp
import numpy
import ipdb

jax.config.update("jax_enable_x64", True)


#def hfun(z):
#    number_of_js = len(z) // 2

#    d_init = z[:number_of_js]
#    d_pert = z[number_of_js:]
#    v1 = jnp.sqrt(d_init)
#    v2 = jnp.sqrt(d_pert)
#    v3 = (v1 - v2) ** 2
#    dphi = 1e-5  # dphi (and N) should probably be passed to hfun as optional arguments for more general CFI types
#    N = 4
#    res = (-4.0 / (N * dphi)**2) * jnp.sum(v3)
#    return res

def hfun(z):
    number_of_js = len(z) // 2

    d_init = z[:number_of_js]
    d_pert = z[number_of_js:]
    v1 = jnp.sqrt(d_init * d_pert)
    v2 = d_init + d_pert - 2 * v1
    dphi = 1e-5  # dphi (and N) should probably be passed to hfun as optional arguments for more general CFI types
    N = 4
    res = (-4.0 / (N * dphi) ** 2) * jnp.sum(v2)
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
    return (H + H.T) / 2.0


def combinemodels_jax(Cres, Gres, Hres):
    return G_combine(Cres, Gres), H_combine(Cres, Gres, Hres)
