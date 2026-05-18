from .create_trsp_solver import create_trsp_solver
from .pounders import pounders
from .pounders_concurrent import pounders as pounders_concurrent


def run_pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, objective=None, concurrent=False):
    r"""
    Run |pounders| on the optimization problem specified by the given
    arguments.

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
    :param objective: ``dict`` that defines objective function :math:`f` to use.
        Set to ``None`` to use the default
        :py:func:`ibcdfo.pounders.h_leastsquares` objective function.

        * **hfun** - Outer function :math:`\hfun` that maps given
          :math:`\Ffun(\psp)` to scalars for minimization
        * **combinemodels** - Function that maps the linear and quadratic terms
          from the models of :math:`\Ffun` into a single quadratic model
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
    # High-level interface that 99% of users should use.  Because of this, users
    # should call it as ibcdfo.run_pounders.
    #
    # The low-level routine should only be called by power-users as
    # ibcdfo.pounders.run_expert_mode.

    # ----- HARDCODED VALUES
    SPSOLVER_MINQ5 = 2

    # ----- CHOOSE DEFAULT VALUES ON-BEHALF OF USERS
    # All non-power users should use the MINQ5 TRSP, which implies that all
    # other choices of TRSP solver require the use of the low-level interface.
    if objective is None:
        objective = {}
    assert "spsolver" not in objective
    objective["spsolver"] = create_trsp_solver(SPSOLVER_MINQ5)

    # ----- OPTIMIZE!
    if concurrent:
        return pounders_concurrent(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Options=objective)

    return pounders(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, Options=objective)
