import copy

from .constants import TRSP_SOLVER_MINQ5
from .create_trsp_solver import create_trsp_solver
from .general_h_funs import h_leastsquares, combine_leastsquares
from .pounders import pounders
from .pounders_concurrent import pounders as pounders_concurrent


def run_user_friendly(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, hfun=None, Prior=None, concurrent=False):
    r"""
    Run |pounders| on the optimization problem specified by the given arguments.

    :param Ffun:    Function that returns :math:`\Ffun(\psp)` as
        **m**-element NumPy array for given :math:`\psp`
    :param X_0:     **n**-element 1D NumPy array that specifies the
        initial point, which must satisfy the boundary constraints
    :param n:       Dimension (number of continuous, real-valued input variables)
    :param nf_max:  Maximum number of function evaluations (> **n** + 1 if
        **Prior** not provided or **nfs** = 0; :math:`\ge` **n** + 1, if **nfs** > 0)
    :param g_tol:   Tolerance for the 2-norm of the model gradient
    :param delta_0: Positive initial trust region radius
    :param m:       Dimension of output of **Ffun** (number of component functions)
    :param Low:     **n**-element 1D NumPy array of lower bounds
    :param Upp:     **n**-element 1D NumPy array of upper bounds
    :param hfun: ``dict`` that defines objective function :math:`f` to use.
        Set to ``None`` to use the default
        :py:func:`ibcdfo.pounders.h_leastsquares` hfun function.

        * **hfun** - Outer function :math:`\hfun` that maps the value
          :math:`\Ffun(\psp)` computed with **Ffun** to scalars for minimization
        * **combinemodels** - Matching function that maps the linear and
          quadratic terms from the models of :math:`\Ffun` into a single
          quadratic model

    :param Prior:   ``dict`` describing past evaluations of **Ffun**.  Set to ``None``
        to run optimization assuming no past evaluations. A nonempty **Prior** must
        contain entries:

        * **nfs** - Number of past function evaluations
        * **X_init** - **nfs** :math:`\times` **n** NumPy array of distinct
          points :math:`\psp_i`
        * **F_init** - **nfs** :math:`\times` **m** NumPy array of values
          :math:`\Ffun(\psp_i)` computed with **Ffun**
        * **xk_in** - Zero-based index into **X_init** and **F_init** that
          corresponds to the point and value to use as the initial point for
          optimization. Note that if **Prior** is nonempty, **X_init[xk_in]**
          and **X_0** must be identical and still satisfy the boundary
          constraints.

    :param concurrent: Set to ``True`` if **Ffun** is parallelized and you would
        like |pounders| to make use of that potential performance increase.

    :return:
        * **X** - :math:`k \times` **n** NumPy array containing all points of
          evaluation :math:`\psp_i` (including those provided in **Prior**) in
          the order in which they were evaluated, where **nfs** :math:`< k \le`
          **nf_max** + **nfs**.
        * **F** - :math:`k \times` **m** NumPy array of values :math:`\Ffun(\psp_i)`
          computed with **Ffun** at all :math:`\psp_i` in  **X** and provided
          with matching ordering.
        * **hF** - :math:`k`-element 1D NumPy array of composed values
          :math:`\hfun(\Ffun(\psp_i))` computed with **hfun** and **Ffun** at
          all :math:`\psp_i` in  **X** and provided with matching ordering.
        * **flag** - Termination criteria flag (see general |pounders| documentation)
        * **xk_in** - Zero-based index of point in **X** representing the
          incumbent at termination (approximate local minimizer if ``flag=0``)
    """
    # ----- CHOOSE DEFAULT VALUES ON-BEHALF OF USERS
    # All non-power users should use the MINQ5 TRSP, which implies that all
    # other choices of TRSP solver require the use of the low-level interface.
    if hfun is None:
        Options = {"hfun": h_leastsquares, "combinemodels": combine_leastsquares}
    else:
        Options = copy.deepcopy(hfun)
    if set(Options) != {"hfun", "combinemodels"}:
        raise ValueError("Error: Invalid hfun configuration")
    Options["spsolver"] = create_trsp_solver(TRSP_SOLVER_MINQ5)

    # ----- OPTIMIZE!
    # Let POUNDERS error check the majority of the arguments.
    if concurrent:
        return pounders_concurrent(
            Ffun=Ffun,
            X_0=X_0,
            n=n,
            nf_max=nf_max,
            g_tol=g_tol,
            delta_0=delta_0,
            m=m,
            Low=Low,
            Upp=Upp,
            Options=Options,
            Prior=Prior,
            Model=None,
        )

    return pounders(
        Ffun=Ffun,
        X_0=X_0,
        n=n,
        nf_max=nf_max,
        g_tol=g_tol,
        delta_0=delta_0,
        m=m,
        Low=Low,
        Upp=Upp,
        Options=Options,
        Prior=Prior,
        Model=None,
    )
