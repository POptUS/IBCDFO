import numpy as np

import jax
import jax.numpy as jnp
import jaxnp_hash.numpy as jnp_h
import jaxnp_hash as jnph
jax.config.update("jax_enable_x64", True)

"""
Jax-hash versions of the hand-coded outer functions h in
general_nonsmooth_h_funs.py / create_*_hfun.py.

Each of these is just the ordinary (smooth-except-for-max/min/abs) math for the
corresponding hand-coded hfun, wrapped with jnph.h_fun so that jaxnp_hash traces the
max/min/maximum/minimum/abs calls and derives the branch hash automatically instead of
it being hand-derived. There is no jax version of h_quantile: it needs an order
statistic (2nd-smallest of the squared values), and jaxnp_hash's numpy shim only
overrides max/min/maximum/minimum/sum/abs.
"""

_TOL = 1e-8


def _make_h_one_norm_jax():
    def f(z):
        return jnp_h.sum(jnp_h.abs(z))

    return jnph.h_fun(f, tol=_TOL)


h_one_norm_jax = _make_h_one_norm_jax()


def _make_h_pw_maximum_jax():
    def f(z):
        return jnp_h.max(z)

    return jnph.h_fun(f, tol=_TOL)


h_pw_maximum_jax = _make_h_pw_maximum_jax()


def _make_h_pw_maximum_squared_jax():
    def f(z):
        return jnp_h.max(z**2)

    return jnph.h_fun(f, tol=_TOL)


h_pw_maximum_squared_jax = _make_h_pw_maximum_squared_jax()


def _make_h_pw_minimum_jax():
    def f(z):
        return jnp_h.min(z)

    return jnph.h_fun(f, tol=_TOL)


h_pw_minimum_jax = _make_h_pw_minimum_jax()


def _make_h_pw_minimum_squared_jax():
    def f(z):
        return jnp_h.min(z**2)

    return jnph.h_fun(f, tol=_TOL)


h_pw_minimum_squared_jax = _make_h_pw_minimum_squared_jax()


def _make_h_max_plus_quadratic_violation_penalty_jax():
    alpha = 0.0

    def f(z):
        p1 = z.shape[0] - 1
        h1 = jnp_h.max(z[:p1])
        h2 = alpha * jnp_h.sum(jnp_h.maximum(z[p1:], 0.0) ** 2)
        return h1 + h2

    return jnph.h_fun(f, tol=_TOL)


h_max_plus_quadratic_violation_penalty_jax = _make_h_max_plus_quadratic_violation_penalty_jax()


def create_piecewise_quadratic_hfun_jax(Qs, zs, cs):
    Qs = jnp.asarray(Qs)
    zs = jnp.asarray(zs)
    cs = jnp.asarray(np.squeeze(cs))

    def f(z):
        n, J = zs.shape
        vals = jnp.stack([jnp.dot(jnp.dot((z - zs[:, j]), Qs[:, :, j]), (z - zs[:, j])) + cs[j] for j in range(J)])
        return jnp_h.max(vals)

    return jnph.h_fun(f, tol=_TOL)


def create_censored_L1_loss_hfun_jax(C, D):
    C = jnp.asarray(np.asarray(C).flatten())
    D = jnp.asarray(np.asarray(D).flatten())

    def f(z):
        return jnp_h.sum(jnp_h.abs(D - jnp_h.maximum(z, C)))

    return jnph.h_fun(f, tol=_TOL)


def _make_h_max_gamma_over_KY_jax():
    KY_jax = jnp.array(np.linspace(0.10, 0.60, 11))

    def f(z_in):
        vals = z_in / KY_jax
        return jnp_h.max(vals)

    return jnph.h_fun(f, tol=_TOL)


h_max_gamma_over_KY_jax = _make_h_max_gamma_over_KY_jax()
