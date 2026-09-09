import sys
import warnings

import numpy as np

from .constants import TRSP_SOLVER_SIMPLE, TRSP_SOLVER_MINQ5, WARNING_SIMPLE_TRSP
from .._get_minq_installation import get_minq_installation
from .bqmin import bqmin


def create_trsp_solver(spsolver):
    r"""
    Create a Python function that solves the bound-constrained trust-region
    subproblem (TRSP)

    .. math::
        \argmin_{\svec \in \R^{\np}}  \gvec^T \svec + \frac{1}{2}\svec^T H \svec

    such that

    .. math::
        Low_j \leq s_j \le Upp_j

    for all components :math:`s_j` of :math:`\svec`.

    :param spsolver:
        * ``ibcdfo.pounders.TRSP_SOLVER_MINQ5`` - Arnold Neumaier's minq5 solver
        * ``ibcdfo.pounders.TRSP_SOLVER_MINQ8`` - Arnold Neumaier's minq8 solver
    :return: Python function with the interface

        .. code:: python

            Xsp, mdec, found_solution = solve_trsp(H, g, Low, Upp)

        where

        * **H** is an :math:`\np \times \np` NumPy array that provides the
          (symmetric) Hessian of the objective function,
        * **g** is an :math:`\np`-element NumPy array that provides the gradient
          of the objective function,
        * **Low** and **Upp** are :math:`\np`-element NumPy arrays that specify
          the bound constraints,
        * **Xsp** is the subproblem solution as an :math:`\np`-element NumPy
          array,
        * **mdec** is the value of the subproblem objective function at
          the solution as a real scalar, and
        * **found_solution** is ``True`` if a solution was found that should be
          acceptable for |pounders|'s purposes; ``False``, otherwise.
    """
    if spsolver == TRSP_SOLVER_SIMPLE:
        # Since this solver is for testing/debugging only, we do not mention it
        # in the documentation nor do we put it in the package's public
        # interface.
        warnings.warn(WARNING_SIMPLE_TRSP)

        def __bqmin_wrapper(H, g, Low, Upp):
            # Assume that solver error checks its arguments thoroughly.
            Xsp, mdec = bqmin(H, g, Low, Upp)
            return Xsp, mdec, True

        return __bqmin_wrapper

    elif spsolver == TRSP_SOLVER_MINQ5:
        # Implement in such a way that users that would like to use a non-MINQ
        # solver don't have to install MINQ.  In other words, allow MINQ to be
        # an *optional* external dependence.
        required_minq_SHA, minq_installation = get_minq_installation()
        if not minq_installation["is_valid"]:
            msg = f"Please set MINQ clone to git commit {required_minq_SHA}.\nSee User Guide (https://ibcdfo.readthedocs.io) for more information and instructions."
            sys.exit(msg)
        from minqsw import minqsw

        def __minq5_wrapper(H, g, Low, Upp):
            # Assume that solver error checks its arguments thoroughly.
            n = H.shape[0]
            Xsp, mdec, minq_err, _ = minqsw(0, g, H, Low.T, Upp.T, 0, np.zeros((n, 1)))
            Xsp = np.atleast_1d(np.squeeze(Xsp))
            mdec = float(np.squeeze(mdec))

            # Continuous function restricted to (compact) k-cell.
            assert minq_err != 1
            # TODO: Since we are solving a subproblem, there is likely no sense
            # in spending an excessive number of iterations seeking a slightly
            # better approximation to the solution.  But, it might be useful for
            # developers/power users to be able to identify when the budget
            # limit is reached.  Once we have improved logging, print debug
            # messages at high verbosity levels if minq_err == 99?  Since we are
            # returning a boolean, all logging would have to be done by this
            # wrapper layer.
            # assert minq_err != 99
            return Xsp, mdec, (minq_err >= 0)

        return __minq5_wrapper

    raise ValueError(f"Unknown trust-region subproblem solver: {spsolver}")
