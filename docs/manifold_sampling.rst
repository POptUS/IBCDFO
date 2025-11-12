Manifold Sampling
=================

The Manifold Sampling code solves the problem

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp))

where :math:`\Ffun` is a blackbox function mapping from :math:`\R^{\np}` to
:math:`\R^{\nd}`, and :math:`\hfun` is a nonsmooth function mapping from
:math:`\R^{\nd}` to :math:`\R`.

More details can be found in :cite:t:`MSP_2021`, :cite:t:`MSP_2018`, and
:cite:t:`MSP_2016`.

Programmatic Interface
----------------------
All |pounders| implementations return a termination criteria flag.  The
interpretation of the value of the flag is identical across implementations
and possible values are

    * norm of final model gradient if optimization was successful
    * -1 - Terminated in error
    * 0 - ``nf_max`` function evaluations were performed

The programmatic interface is generally maintained identical between all
implementations.  Nevertheless, we provide the interface for each implementation
to provide language-specific descriptions.

Python
^^^^^^
.. autofunction:: ibcdfo.manifold_sampling.manifold_sampling_primal

|matlab|
^^^^^^^^
.. mat:autofunction:: manifold_sampling.m.manifold_sampling_primal
