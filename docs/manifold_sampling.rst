Manifold Sampling
=================

General Description
-------------------
Manifold Sampling is a derivative-free trust-region method for solving composite
nonsmooth optimization problems of the form

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp))

where:

* :math:`\Ffun:\R^{\np} \rightarrow \R^{\nd}` is a black-box vector-valued
  simulation or residual function, and
* :math:`\hfun:\R^{\nd} \rightarrow \R` is a nonsmooth outer function,
  typically piecewise linear, piecewise quadratic, or formed from max, min,
  absolute value, or selection operations.

In the programmatic interface, the input to ``hfun`` is often denoted
:math:`\zvec`, representing the intermediate vector
:math:`\zvec = \Ffun(\psp)`.

Typical applications assume that :math:`\Ffun` is continuous and locally smooth
enough that accurate trust-region surrogate models can be constructed, although
derivatives of :math:`\Ffun` need not be available.

This structure arises naturally in robust regression, minimax design,
simulation-based calibration, and optimization with embedded nonsmooth merit
functions.

The method builds local surrogate models for components of :math:`\Ffun` and
exploits the piecewise structure of :math:`\hfun` by identifying locally active
manifolds: smooth pieces :math:`\hfun` that are
active near the current iterate. At each iteration, a trust-region subproblem is
solved on a model of the currently relevant manifolds, producing a step that
balances local improvement and model accuracy.

Unlike generic nonsmooth methods, manifold sampling leverages the composite
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

  ``flag > 0``
      Successful termination. The returned value is the final stationarity
      measure (:math:`\chi_k`), indicating the trust-region radius fell
      below the minimum threshold.

  ``0``
      ``nf_max`` function evaluations were performed.

  ``-1``
      Model construction failed (empty or invalid local model).

  ``-2``
      Trust-region subproblem failed, likely due to an unbounded or poorly
      scaled affine-envelope subproblem.

The programmatic interface is generally maintained identically across all
implementations. Nevertheless, we provide the interface for each implementation
to supply language-specific descriptions.

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
