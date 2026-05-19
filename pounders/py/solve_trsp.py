import sys
from functools import lru_cache

import numpy as np

from .._get_minq_installation import get_minq_installation
from .bqmin import bqmin


@lru_cache(maxsize=1)
def _get_minqsw():
    required_minq_SHA, minq_installation = get_minq_installation()

    if not minq_installation["is_valid"]:
        msg = f"Please set MINQ clone to git commit {required_minq_SHA}.\nSee User Guide (https://ibcdfo.readthedocs.io) for more information and instructions."
        sys.exit(msg)

    from minqsw import minqsw

    return minqsw


def solve_trsp(H, G, Low, Upp, xk, delta, spsolver, n):
    """
    Solve the bound-constrained trust-region subproblem.

    min  G.T * s + 0.5 * s.T * H * s
    s.t. max(Low - xk, -delta) <= s <= min(Upp - xk, delta)
    """
    Lows = np.maximum(Low - xk, -delta * np.ones(np.shape(Low)))
    Upps = np.minimum(Upp - xk, delta * np.ones(np.shape(Upp)))

    if spsolver == 1:
        Xsp, mdec = bqmin(H, G, Lows, Upps)
        return Xsp, mdec, 0

    if spsolver == 2:
        minqsw = _get_minqsw()
        Xsp, mdec, minq_err, _ = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        if minq_err < 0:
            return Xsp, mdec, -4
        return Xsp, mdec, 0

    raise ValueError(f"Unknown trust-region subproblem solver: {spsolver}")
