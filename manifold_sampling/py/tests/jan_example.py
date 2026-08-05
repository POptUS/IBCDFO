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


def _hfun_jax(f):
    return jnph.h_fun(f, tol=_TOL)


h_one_norm_jax = _hfun_jax(lambda z: jnp_h.sum(jnp_h.abs(z)))
h_pw_maximum_jax = _hfun_jax(lambda z: jnp_h.max(z))
h_pw_maximum_squared_jax = _hfun_jax(lambda z: jnp_h.max(z**2))
h_pw_minimum_jax = _hfun_jax(lambda z: jnp_h.min(z))
h_pw_minimum_squared_jax = _hfun_jax(lambda z: jnp_h.min(z**2))

# alpha=0.0 zeroes the quadratic-violation-penalty term's contribution to the value, but
# keeping the term in place keeps the hash structure comparable to the hand-coded version.
_ALPHA = 0.0
h_max_plus_quadratic_violation_penalty_jax = _hfun_jax(
    lambda z: jnp_h.max(z[: z.shape[0] - 1]) + _ALPHA * jnp_h.sum(jnp_h.maximum(z[z.shape[0] - 1 :], 0.0) ** 2)
)


def create_piecewise_quadratic_hfun_jax(Qs, zs, cs):
    Qs = jnp.asarray(Qs)
    zs = jnp.asarray(zs)
    cs = jnp.asarray(np.squeeze(cs))
    J = zs.shape[1]

    return _hfun_jax(lambda z: jnp_h.max(jnp.stack([jnp.dot(jnp.dot((z - zs[:, j]), Qs[:, :, j]), (z - zs[:, j])) + cs[j] for j in range(J)])))


def create_censored_L1_loss_hfun_jax(C, D):
    C = jnp.asarray(np.asarray(C).flatten())
    D = jnp.asarray(np.asarray(D).flatten())

    return _hfun_jax(lambda z: jnp_h.sum(jnp_h.abs(D - jnp_h.maximum(z, C))))


_KY_jax = jnp.array(np.linspace(0.10, 0.60, 11))
h_max_gamma_over_KY_jax = _hfun_jax(lambda z_in: jnp_h.max(z_in / _KY_jax))
