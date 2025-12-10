|pounders|
==========

The Practical Optimization Using No Derivatives and Exploiting Recognized
Structure method, better known as |pounders|, minimizes a sum of squares of
blackbox (''zeroth-order'') functions, solving

.. math::
   \min_{\psp \in \R^{\np}} \left\{f(\psp)=\sum_{i=1}^{\nd} F_i(\psp)^2\right\}

subject to

.. math::
    Low_j \le \pspcomp{j} \le Upp_j, j=1,...,\np,

where 

* where :math:`\Ffun` is the vector-valued, user-provided blackbox function,
* :math:`Low` is a user-provided boundary constraint that permits values of
  :math:`-\infty` to specify that the problem is unconstrained for the
  associated parameter, and
* :math:`Upp` is a user-provided boundary constraint that permits values of
  :math:`+\infty` to specify that the problem is unconstrained for the
  associated parameter.

The algorithm will not evaluate :math:`\Ffun` outside of these bounds, but it is
possible to take advantage of function values at infeasible :math:`\psp` if
these are passed initially through ``(X_init,F_init)``.  In each iteration, the
algorithm forms a set of quadratic models interpolating the functions in
:math:`\Ffun` and minimizes an associated scalar-valued model within an
infinity-norm trust region.

Optionally, users can specify a custom outer-function :math:`\hfun` that, like
:math:`f`, maps the elements of :math:`\Ffun` to a scalar value
:math:`\hfun(\Ffun(\psp))` for minimization. Users must also provide a "combine
models" function that |pounders| uses to map the linear and quadratic terms from
the models of :math:`\Ffun` into a single quadratic model.

For more detailed information please refer to :cite:t:`POUNDERS_TAO_2017`.  A
brief description can also be found in :cite:t:`UNEDF0_2010`.

Programmatic Interface
----------------------
All |pounders| implementations return a termination criteria flag.  The
interpretation of the value of the flag is identical across implementations
and possible values are

* 0 - normal termination because norm of :math:`\gradf(\psp)` at final
  :math:`\psp` satisfied user-provided gradient tolerance,
* > 0 - exceeded the maximum number evals and the value is the 2-norm of
  :math:`\gradf` at final :math:`\psp`
* -1 - input was fatally incorrect (error message shown)
* -2 - a valid model produced ``X[nf] == X[xk_in]`` or ``(mdec == 0, hF[nf] == hF[xk_in])``
* -3 - a ``NaN`` was encountered
* -4 - error in TRSP Solver
* -5 - unable to get model improvement with current parameters
* -6 - delta has reached delta_min with a valid model

The programmatic interface is generally maintained identical between all
implementations.  Nevertheless, we provide the interface for each implementation
to provide language-specific descriptions.

Python
^^^^^^
.. autofunction:: ibcdfo.run_pounders

.. autofunction:: ibcdfo.pounders.h_leastsquares

|matlab|
^^^^^^^^
.. mat:autofunction:: pounders.m.pounders
