import platform
import sys

import ibcdfo
import jax
import numpy as np
import scipy
from calfun import calfun
from dfoxs import dfoxs
from ibcdfo.manifold_sampling import h_max_gamma_over_KY

from jan_example import h_max_gamma_over_KY_jax

print("###GAMMA_EXAMPLE_DIAG### env: python=%s platform=%s" % (sys.version.split()[0], platform.platform()), flush=True)
print(
    "###GAMMA_EXAMPLE_DIAG### env: numpy=%s scipy=%s jax=%s jaxlib=%s jax_enable_x64=%s backend=%s devices=%s"
    % (np.__version__, scipy.__version__, jax.__version__, jax.lib.__version__, jax.config.jax_enable_x64, jax.default_backend(), jax.devices()),
    flush=True,
)

dfo = np.loadtxt("dfo.dat")

PROBS_TO_SOLVE = [16, 33]
SUBPROB_SWITCH = "linprog"
NF_MAX = 150


def _report_diff(prob_row, name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f"###GAMMA_EXAMPLE_DIAG### row {prob_row} {name}: SHAPE MISMATCH old={a.shape} jax={b.shape}", flush=True)
        return False
    close = np.allclose(a, b, rtol=1e-6, atol=1e-8)
    diff = np.abs(a - b)
    max_diff = float(diff.max()) if diff.size else 0.0
    idx = np.unravel_index(np.argmax(diff), diff.shape) if diff.size else None
    print(
        f"###GAMMA_EXAMPLE_DIAG### row {prob_row} {name}: allclose={close} max_abs_diff={max_diff:.6e} at index={idx}",
        flush=True,
    )
    if idx is not None and max_diff > 0:
        print(f"###GAMMA_EXAMPLE_DIAG### row {prob_row} {name}: old{idx}={a[idx]!r} jax{idx}={b[idx]!r}", flush=True)
    return close


all_passed = True

for prob_row in PROBS_TO_SOLVE:
    nprob, n, m, factor_power = dfo[prob_row, :]
    n = int(n)
    m = int(m)
    LB = -np.inf * np.ones(n)
    UB = np.inf * np.ones(n)
    x0 = dfoxs(n, nprob, 10**factor_power)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=2)[1]
        assert len(out) == m, "Incorrect output dimension"
        return np.squeeze(out)

    X_old, F_old, h_old, xkin_old, flag_old = ibcdfo.run_MSP(h_max_gamma_over_KY, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)
    X_jax, F_jax, h_jax, xkin_jax, flag_jax = ibcdfo.run_MSP(h_max_gamma_over_KY_jax, Ffun, x0, LB, UB, NF_MAX, SUBPROB_SWITCH)

    print(f"###GAMMA_EXAMPLE_DIAG### row {prob_row} (prob {int(nprob)}): xkin_old={xkin_old} xkin_jax={xkin_jax} flag_old={flag_old} flag_jax={flag_jax}", flush=True)

    x_ok = _report_diff(prob_row, "X", X_old, X_jax)
    f_ok = _report_diff(prob_row, "F", F_old, F_jax)
    h_ok = _report_diff(prob_row, "h_msp", h_old, h_jax)

    if x_ok and f_ok and h_ok:
        print(f"dfo row {prob_row} (prob {int(nprob)}): hand-coded and jax h_max_gamma_over_KY agree")
    else:
        print(f"###GAMMA_EXAMPLE_DIAG### row {prob_row} (prob {int(nprob)}): DIVERGED (see max_abs_diff lines above)", flush=True)
        all_passed = False

assert all_passed, "See ###GAMMA_EXAMPLE_DIAG### lines above for details on which row/array diverged"
