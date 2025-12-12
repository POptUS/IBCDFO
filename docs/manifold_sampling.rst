Manifold Sampling
=================

.. todo::

    * Does this method assume certain characteristics of :math:`\Ffun`?
    * The notation here is in terms of composing :math:`\hfun` and :math:`\Ffun`,
      which matches the notation in |pounders|.  However, manifold sampling
      inline docs mention that hfuns work with :math:`\zvec`.  Reconcile
      notation.
    * I suspect that a general discussion of manifolds and h functions would be
      useful here.  An upfront discussion might also allow for significant
      simplification of the inline documentation of ``run_MSP`` and the ``h_*``
      functions.

General Description
-------------------
The Manifold Sampling method solves the problem

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp))

where :math:`\Ffun` is a blackbox function mapping from :math:`\R^{\np}` to
:math:`\R^{\nd}`, and :math:`\hfun` is a nonsmooth function mapping from
:math:`\R^{\nd}` to :math:`\R`.

More details can be found in :cite:t:`MSP_2024`, :cite:t:`MSP_2021`,
:cite:t:`MSP_2018`, and :cite:t:`MSP_2016`.

Programmatic Interface
----------------------
Status Code
^^^^^^^^^^^
All Manifold Sampling implementations return a termination criteria flag.  The
interpretation of the value of the flag is identical across implementations and
possible values are

    * norm of final model gradient if optimization was successful
    * -1 - Terminated in error
    * 0 - ``nf_max`` function evaluations were performed

The programmatic interface is generally maintained identical between all
implementations.  Nevertheless, we provide the interface for each implementation
to provide language-specific descriptions.

Python
^^^^^^
.. autofunction:: ibcdfo.run_MSP

|matlab|
^^^^^^^^
.. mat:autofunction:: manifold_sampling.m.manifold_sampling_primal

:math:`\hfun` Functions
^^^^^^^^^^^^^^^^^^^^^^^
.. todo::

    * The set of functions available in |matlab| and Python are **not**
      equal.

The following :math:`\hfun` functions are available for use with both the Python
and |matlab| implementations of Manifold Sampling.  While they are presented
through their integration into the Python package, the documentation is valid
for the |matlab| version of these functions, which are located in
``manifold_sampling/m/general_smooth_h_funs``.

.. autofunction:: ibcdfo.manifold_sampling.h_one_norm
.. autofunction:: ibcdfo.manifold_sampling.h_censored_L1_loss
.. autofunction:: ibcdfo.manifold_sampling.h_pw_minimum
.. autofunction:: ibcdfo.manifold_sampling.h_pw_minimum_squared
.. autofunction:: ibcdfo.manifold_sampling.h_pw_maximum
.. autofunction:: ibcdfo.manifold_sampling.h_pw_maximum_squared
.. autofunction:: ibcdfo.manifold_sampling.h_piecewise_quadratic
.. autofunction:: ibcdfo.manifold_sampling.h_quantile
.. autofunction:: ibcdfo.manifold_sampling.h_max_gamma_over_KY
