import branch_extended_AD.numpy as jnp_h
import jax
import jax.numpy as jnp
import numpy as np
from branch_extended_AD.integrations.ibcdfo import h_fun

jax.config.update("jax_enable_x64", True)

"""
Jax-hash versions of the hand-coded outer functions h in
general_nonsmooth_h_funs.py / create_*_hfun.py.

Each of these is just the ordinary (smooth-except-for-max/min/abs) math for the
corresponding hand-coded hfun, wrapped with h_fun (branch_extended_AD.integrations.ibcdfo)
so that branch_extended_AD traces the max/min/maximum/minimum/abs calls and derives the
branch hash automatically instead of it being hand-derived. There is no jax version of
h_quantile: it needs an order statistic (2nd-smallest of the squared values), and
branch_extended_AD's numpy shim only overrides max/min/maximum/minimum/sum/abs.
"""

_TOL = 1e-8


def _hfun_jax(f, tol=_TOL):
    # atol=tol, rtol=0.0 reproduces the old single-`tol` API's absolute-only tie
    # tolerance exactly (the new h_fun's "local" tol_mode uses atol + rtol*|reference|,
    # same formula as the hand-coded side's `_activities_and_inds(atol=, rtol=)`).
    return h_fun(f, atol=tol, rtol=0.0)


def _one_norm(z):
    return jnp_h.sum(jnp_h.abs(z))


def _pw_maximum(z):
    return jnp_h.max(z)


def _pw_maximum_squared(z):
    return jnp_h.max(z**2)


def _pw_minimum(z):
    return jnp_h.min(z)


def _pw_minimum_squared(z):
    return jnp_h.min(z**2)


# alpha=0.0 zeroes the quadratic-violation-penalty term's contribution to the value, but
# keeping the term in place keeps the hash structure comparable to the hand-coded version.
_ALPHA = 0.0


def _max_plus_quadratic_violation_penalty(z):
    return jnp_h.max(z[: z.shape[0] - 1]) + _ALPHA * jnp_h.sum(jnp_h.maximum(z[z.shape[0] - 1 :], 0.0) ** 2)


# Module-level singletons built at default tolerance, for backward compatibility
h_one_norm_jax = _hfun_jax(_one_norm)
h_pw_maximum_jax = _hfun_jax(_pw_maximum)
h_pw_maximum_squared_jax = _hfun_jax(_pw_maximum_squared)
h_pw_minimum_jax = _hfun_jax(_pw_minimum)
h_pw_minimum_squared_jax = _hfun_jax(_pw_minimum_squared)
h_max_plus_quadratic_violation_penalty_jax = _hfun_jax(_max_plus_quadratic_violation_penalty)

# Mapping of simple hfun names to their builder functions for parameterized tol sweeps
SIMPLE_JAX_HFUN_BUILDERS = {
    "h_one_norm": _one_norm,
    "h_pw_maximum": _pw_maximum,
    "h_pw_maximum_squared": _pw_maximum_squared,
    "h_pw_minimum": _pw_minimum,
    "h_pw_minimum_squared": _pw_minimum_squared,
    "h_max_plus_quadratic_violation_penalty": _max_plus_quadratic_violation_penalty,
}


def create_piecewise_quadratic_hfun_jax(Qs, zs, cs, tol=_TOL):
    Qs = jnp.asarray(Qs)
    zs = jnp.asarray(zs)
    cs = jnp.asarray(np.squeeze(cs))

    # Vectorized over the J pieces (single fused dispatch) instead of a Python loop over
    # J individual jnp.dot calls -- each un-jitted dispatch costs ~10ms, so for J~90
    # pieces (e.g. dfo rows 0/1) the loop version costs >1s per hfun call.
    def f(z):
        diffs = z[:, None] - zs
        quad = jnp.einsum("mj,mnj,nj->j", diffs, Qs, diffs)
        return jnp_h.max(quad + cs)

    return _hfun_jax(f, tol=tol)


def create_censored_L1_loss_hfun_jax(C, D, tol=_TOL):
    C = jnp.asarray(np.asarray(C).flatten())
    D = jnp.asarray(np.asarray(D).flatten())

    return _hfun_jax(lambda z: jnp_h.sum(jnp_h.abs(D - jnp_h.maximum(z, C))), tol=tol)


_KY_jax = jnp.array(np.linspace(0.10, 0.60, 11))


def _max_gamma_over_KY(z_in):
    return jnp_h.max(z_in / _KY_jax)


h_max_gamma_over_KY_jax = _hfun_jax(_max_gamma_over_KY)

# Also add to SIMPLE_JAX_HFUN_BUILDERS for consistency (though test_gamma_example.py imports directly)
SIMPLE_JAX_HFUN_BUILDERS["h_max_gamma_over_KY"] = _max_gamma_over_KY
