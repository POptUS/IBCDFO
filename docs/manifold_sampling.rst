.. _msp_section:

|manifold_sampling|
===================
General Description
-------------------
Manifold Sampling is a derivative-free trust-region method for solving composite
nonsmooth optimization problems of the form

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp))

where:

* :math:`\Ffun:\R^{\np} \rightarrow \R^{\nd}` is a black-box vector-valued
  function (e.g., a simulation output or a vector of residuals), and
* :math:`\hfun:\R^{\nd} \rightarrow \R` is a nonsmooth function, expressible
  as a continuous selection. Examples of continuous selections include
  piecewise linear functions, piecewise quadratic functions or functions formed from max, min
  or absolute value operations.

In the programmatic interface, the input to ``hfun`` is often denoted
:math:`\zvec`, representing the intermediate vector
:math:`\zvec = \Ffun(\psp)`.

Analyses assume that :math:`\Ffun` is continuous and locally sufficiently smooth
so that accurate trust-region surrogate models can be constructed. However,
in practice, these assumptions do not need to be explicitly checked and
derivatives of :math:`\Ffun` certainly need not be available.

The composite structure :math:`\hfun(\Ffun(\psp))` arises naturally in robust regression, minimax design,
simulation-based calibration, and optimization with embedded nonsmooth merit
functions.

Manifold Sampling builds local surrogate models for components of :math:`\Ffun` and
exploits the piecewise structure of :math:`\hfun` by identifying locally active
manifolds, that is, smooth pieces :math:`\hfun` that are
active near the current incumbent. At each iteration, a trust-region subproblem is
solved on a model derived in part from the currently active manifolds, producing a step that
balances local improvement and model accuracy.

Unlike generic nonsmooth methods, manifold sampling deliberately leverages the composite
structure :math:`\hfun(\Ffun(\psp))`, which can significantly improve practical
performance when evaluations of :math:`\Ffun` are expensive.

For algorithmic details, see :cite:t:`MSP_2016`, :cite:t:`MSP_2018`,
:cite:t:`MSP_2021`, and :cite:t:`MSP_2024`.

Programmatic Interface
----------------------
Status Code
^^^^^^^^^^^
All Manifold Sampling implementations return a termination criteria flag. The
interpretation of the value of the flag is identical across implementations and
possible values are

* 1 - budget of ``nf_max`` function evaluations was exhausted
* 0 - normal termination; the stationary measure (:math:`\chi_k`) and the
  trust-region radius were sufficiently small within tolerance.
* -1 - model construction failed (an empty or invalid local model
  was constructed)
* -2 - trust-region subproblem failed, likely due to an unbounded
  or poorly scaled affine-envelope subproblem
* -6 - the trust-region radius reached its minimum without satisfying the
  stationarity tolerance. This is a common way for a successful run to end; 
  it does not necessarily indicate a failure. The final :math:`\chi_k` is displayed
  when output is enabled. A small :math:`\chi_k` together with the small radius
  indicates a good result. One should inspect ``hF`` as well.

.. note::
   Unlike the norm of the gradient in |pounders|, the stationarity measure in
   manifold sampling :math:`\chi_k` alone is not a reliable indication of how close
   the last incumbent is to a solution. :math:`\chi_k` can be (close to) zero while the
   trust-region radius is still large, in which case the problem could be far from
   solved. For this reason, the flag (1) indicates only that the budget was exhausted
   and does not return the final value of :math:`\chi_k`, like |pounders| does. 

The programmatic interface is generally maintained identically across all
implementations. Nevertheless, we provide the interface for each implementation
to provide language-specific descriptions.

Python
^^^^^^
.. autofunction:: ibcdfo.run_MSP

|matlab|
^^^^^^^^
.. mat:autofunction:: manifold_sampling.m.manifold_sampling_primal

General :math:`\hfun` Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following :math:`\hfun` functions are available for immediate use with the
Python implementation of Manifold Sampling.  With the potential exception of
application-specific functions, these same functions are available for immediate
use with the |matlab| implementation in ``manifold_sampling/m/general_nonsmooth_h_funs``.
While they are presented here through their integration into the Python package,
the documentation is generally valid for the |matlab| version of these functions
as well.  In |matlab| implementations, the ``H0`` argument is also optional.

.. autofunction:: ibcdfo.manifold_sampling.h_one_norm
.. autofunction:: ibcdfo.manifold_sampling.h_pw_minimum
.. autofunction:: ibcdfo.manifold_sampling.h_pw_minimum_squared
.. autofunction:: ibcdfo.manifold_sampling.h_pw_maximum
.. autofunction:: ibcdfo.manifold_sampling.h_pw_maximum_squared
.. autofunction:: ibcdfo.manifold_sampling.h_quantile
.. autofunction:: ibcdfo.manifold_sampling.h_max_gamma_over_KY
.. autofunction:: ibcdfo.manifold_sampling.h_max_plus_quadratic_violation_penalty

Parameterized :math:`\hfun` Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Manifold Sampling permits parameterized :math:`\hfun` functions, which it can
use only once users have chosen a single set of parameter values for formulating
a specific :math:`\hfun` function and, therefore, a single related problem.  As an example, the
following routines can be used to create a single ``hfun`` for a single set of
desired parameter values.  While these routines are presented through their
integration into the Python package, the documentation is valid for the |matlab|
version of these routines, which are located in
``manifold_sampling/m/general_nonsmooth_h_funs``.

.. autofunction:: ibcdfo.manifold_sampling.create_censored_L1_loss_hfun
.. autofunction:: ibcdfo.manifold_sampling.create_piecewise_quadratic_hfun
