import copy
import numpy as np

from .constants import TRSP_SOLVER_MINQ5
from .create_trsp_solver import create_trsp_solver
from .general_h_funs import h_leastsquares, combine_leastsquares
from .pounders import pounders
from .pounders_concurrent import pounders as pounders_concurrent


def run_user_friendly(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, hfun=None, Prior=None, concurrent=False):
    r"""
    Run |pounders| on the optimization problem specified by the given arguments.

    :param Ffun:    Function that returns :math:`\Ffun(\psp)` as :math:`\nd`
        element NumPy array for given :math:`\psp`
    :param X_0:     :math:`\np` element NumPy array that specifies the initial
        point
    :param n:       Dimension (number of continuous, real-valued input variables)
    :param nf_max:  Maximum number of function evaluations (:math:`> \np+1`)
    :param g_tol:   Tolerance for the 2-norm of the model gradient
    :param delta_0: Positive initial trust region radius
    :param m:       Dimension of output of ``Ffun`` (number of component functions)
    :param Low:     :math:`\np` element NumPy array of lower bounds
    :param Upp:     :math:`\np` element NumPy array of upper bounds
    :param hfun: ``dict`` that defines objective function :math:`f` to use.
        Set to ``None`` to use the default
        :py:func:`ibcdfo.pounders.h_leastsquares` hfun function.

        * **hfun** - Outer function :math:`\hfun` that maps given
          :math:`\Ffun(\psp)` to scalars for minimization
        * **combinemodels** - Function that maps the linear and quadratic terms
          from the models of :math:`\Ffun` into a single quadratic model
    :param Prior:   ``dict`` describing  past evaluations of ``Ffun``.  Set to
        ``None`` to run optimization assuming no past evaluations. A nonempty
        **Prior** must contain entries:

        * **nfs** - Number of past function evaluations
        * **X_init** - :math:`\mathrm{nfs} \times \np` NumPy array of points
          :math:`\psp_k`
        * **F_init** - :math:`\mathrm{nfs} \times \nd` NumPy array of values
          :math:`\Ffun(\psp_k)` computed with ``Ffun``
        * **xk_in** -  Zero-based index into ``X_init`` and ``F_init`` that
          corresponds to the point and value to use as initial point for
          optimization. Note that if **Prior** is nonempty, this will override
          the previously specified **X_0**.
    :param concurrent: Set to True if ``Ffun`` is parallelized and you would
        like |pounders| to make use of that potential performance increase.

    :return:
        * **X** - :math:`\mathrm{nf\_max+nfs}\times \np` NumPy array containing
          locations of evaluated points in the order in which they were
          evaluated
        * **F** - :math:`\mathrm{nf\_max+nfs}\times \nd` NumPy array containing
          the function values at ``X`` with matching ordering
        * **hF** - :math:`\mathrm{nf\_max+nfs}\times 1` Composed values
          ``hfun(Ffun(x))`` for evaluated points ``x`` in ``X``
        * **flag** - Termination criteria flag (See general |pounders| documentation)
        * **xk_in** - Zero-based index of point in ``X`` representing
          incumbent at termination (approximate local minimizer if `flag=0`)
    """
    # ----- CHOOSE DEFAULT VALUES ON-BEHALF OF USERS
    # All non-power users should use the MINQ5 TRSP, which implies that all
    # other choices of TRSP solver require the use of the low-level interface.
    if hfun is None:
        Options = {"hfun": h_leastsquares, "combinemodels": combine_leastsquares}
    else:
        Options = copy.deepcopy(hfun)
    if set(Options) != {"hfun", "combinemodels"}:
        raise ValueError("Invalid hfun configuration")
    Options["spsolver"] = create_trsp_solver(TRSP_SOLVER_MINQ5)

    if Prior is None:
        Prior = {"nfs": 0, "X_init": np.full((0, n), np.nan, float), "F_init": np.full((0, m), np.nan, float), "xk_in": 0}

    # ----- OPTIMIZE!
    if concurrent:
        return pounders_concurrent(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Options=Options, Prior=Prior)
    return pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Options=Options, Prior=Prior)
