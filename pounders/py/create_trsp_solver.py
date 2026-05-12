import sys

import numpy as np

from .constants import CRAPPY_TRSP, MINQ5_TRSP
from .._get_minq_installation import get_minq_installation
from .bqmin import bqmin


def create_trsp_solver(spsolver):
    r"""
    Create a Python function that solves the bound-constrained trust-region
    subproblem

    .. math::
        \min_{\svec \in \R^{\np}}  G^T \svec + \frac{1}{2}\svec^T H \svec

    such that

    .. math::
        Low_j \leq s_j \le Upp_j, j=1,...,\np

    for all components :math:`s_j` of :math:`\svec`.

    :param spsolver:
        * ``ibcdfo.pounders.CRAPPY_TRSP`` - Stefan's crappy 10line solver
        * ``ibcdfo.pounders.MINQ5_TRSP`` - Arnold Neumaier's minq5 solver
    :return: Python function with the interface

        .. code:: python

            Xsp, mdec, flag = solve_trsp(H, G, Low, Upp)

        where ...
    """
    if spsolver == CRAPPY_TRSP:

        def __bqmin_wrapper(H, G, Lows, Upps):
            Xsp, mdec = bqmin(H, G, Lows, Upps)
            return Xsp, mdec, 0

        return __bqmin_wrapper
    elif spsolver == MINQ5_TRSP:
        required_minq_SHA, minq_installation = get_minq_installation()
        if not minq_installation["is_valid"]:
            msg = f"Please set MINQ clone to git commit {required_minq_SHA}.\nSee User Guide (https://ibcdfo.readthedocs.io) for more information and instructions."
            sys.exit(msg)

        # Implement in such away that users that would like to use a non-MINQ
        # solver don't have to install MINQ.  In other words, allow MINQ to be
        # an *optional* external dependence.
        from minqsw import minqsw

        def __minq5_wrapper(H, G, Lows, Upps):
            n = H.shape[0]
            Xsp, mdec, minq_err, _ = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
            if minq_err < 0:
                return Xsp, mdec, -4
            return Xsp, mdec, 0

        return __minq5_wrapper

    raise ValueError(f"Unknown trust-region subproblem solver: {spsolver}")
