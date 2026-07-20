import sys
import warnings

import numpy as np

from .constants import TRSP_SOLVER_SIMPLE, TRSP_SOLVER_MINQ5
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
        * ``ibcdfo.pounders.TRSP_SOLVER_MINQ5`` - Arnold Neumaier's minq5 solver
        * ``ibcdfo.pounders.TRSP_SOLVER_MINQ8`` - Arnold Neumaier's minq8 solver
    :return: Python function with the interface

        .. code:: python

            Xsp, mdec, flag = solve_trsp(H, G, Low, Upp)

        where

        * ``H`` is an :math:`\np \times \np` numpy array that provides the
          (symmetric) Hessian of the objective function,
        * ``G`` is an :math:`\np` element numpy array that provides the gradient
          of the objective function,
        * ``Low`` and ``High`` are :math:`\np` element numpy arrays that specify
          the bound constraints,
        * ``Xsp`` is the subproblem solution,
        * ``mdec`` is the value of the subproblem objective function at
          solution, and
        * ``flag`` communicates the termination condition of the solver with a
          negative value indicating failure.
    """
    if spsolver == TRSP_SOLVER_SIMPLE:
        # Since this solver is for testing/debugging only, we do not mention it
        # in the documentation nor do we put it in the package's public
        # interface.
        warnings.warn("The simple TRSP solver should only be used for testing or debugging")

        def __bqmin_wrapper(H, G, Low, Upp):
            Xsp, mdec = bqmin(H, G, Low, Upp)
            return Xsp, mdec, 0

        return __bqmin_wrapper

    elif spsolver == TRSP_SOLVER_MINQ5:
        required_minq_SHA, minq_installation = get_minq_installation()
        if not minq_installation["is_valid"]:
            msg = f"Please set MINQ clone to git commit {required_minq_SHA}.\nSee User Guide (https://ibcdfo.readthedocs.io) for more information and instructions."
            sys.exit(msg)

        # Implement in such away that users that would like to use a non-MINQ
        # solver don't have to install MINQ.  In other words, allow MINQ to be
        # an *optional* external dependence.
        from minqsw import minqsw

        def __minq5_wrapper(H, G, Low, Upp):
            n = H.shape[0]
            Xsp, mdec, minq_err, _ = minqsw(0, G, H, Low.T, Upp.T, 0, np.zeros((n, 1)))
            if minq_err < 0:
                return Xsp, mdec, -4
            # Continuous function restricted to (compact) k-cell.
            assert minq_err != 1
            # TODO: Since we are solving a subproblem, there is likely no sense
            # in spending an excessive number of iterations seeking a slightly
            # better approximation to the solution.  But, it might be useful for
            # developers/power users to be able to identify when the budget
            # limit is reached.  Once we have improved logging, print debug
            # messages at high-verbosity level if minq_err == 99?  Better to
            # return that error code and let POUNDERS log?
            # assert minq_err != 99
            return Xsp, mdec, 0

        return __minq5_wrapper

    raise ValueError(f"Unknown trust-region subproblem solver: {spsolver}")
