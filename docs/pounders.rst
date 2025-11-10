|pounders|: Practical Optimization Using No Derivatives for sums of Squares
===========================================================================

.. note::
    |pounders| implementations do not come with a warranty, are not bug-free,
    and are not for industrial use or public distribution.

.. todo::
    Other references?

Direct requests and bugs to ``wild@lbl.gov``

|pounders| minimizes output from a structured blackbox function :math:`F`,
solving 

.. math::
   \min_{\psp \in \R^{\np}} \left\{f(\psp)=\sum_{i=1}^{\nd} F_i(\psp)^2\right\}

subject to

.. math::
    L_j \le \pspcomp{j} \le U_j, j=1,...,\np,

where 

* :math:`L` is a user-provided boundary constraint that permits values of
  :math:`-\infty` to specify that the problem is unconstrained for the
  associated parameter, and
* :math:`U` is a user-provided boundary constraint that permits values of
  :math:`+\infty` to specify that the problem is unconstrained for the
  associated parameter.

In each iteration, the algorithm forms an interpolating quadratic model of the
function and minimizes it in an infinity-norm trust region.  The algorithm will
not evaluate :math:`F` outside of the bounds indicated by :math:`L,U`.

Optionally, users can provide a set of points :math:`\{\psp_i\}` and the
associated set of values :math:`\{F(\psp_i)\}` that have already been obtained.
It is permissible for these points to lie outside of the boundary constraints
of the problem.

Optionally, users can specify a custom outer-function that, like :math:`f`,
maps the the elements of :math:`F` to a scalar value for minimization. Users
must also provide a "combine models" function that |pounders| uses to map the
linear and quadratic terms from the residual models into a single quadratic
trust-region subproblem (TRSP) model.

All |pounders| implementations return a termination criteria flag.  The
interpretation of the value of the flag is identical across implementations
and possible values are

* 0 - normal termination because norm of :math:`\gradf(\psp)` at final
  :math:`\psp` satistifed user-provided gradient tolerance,
* > 0 - exceeded the maximum number evals and the value is the 2-norm of
  :math:`\gradf` at final :math:`\psp`
* -1 - input was fatally incorrect (error message shown)
* -2 - a valid model produced ``X[nf] == X[xk_in]`` or ``(mdec == 0, hF[nf] == hF[xk_in])``
* -3 - a ``NaN`` was encountered
* -4 - error in TRSP Solver
* -5 - unable to get model improvement with current parameters
* -6 = delta has reached delta_min with a valid model

For more detailed information please refer to :cite:t:`POUNDERS_TAO_2017`.  A
brief description can also be found in :cite:t:`UNEDF0_2010`.

Python
------
.. autofunction:: ibcdfo.pounders.pounders

|matlab|
--------
.. todo::

    * Add MATLAB public API docs

..
    .. mat:currentmodule:: pounders.m
    .. mat:autofunction:: pounders
