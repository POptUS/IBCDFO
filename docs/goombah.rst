|goombah|
=========
General Description
-------------------
This code solves the problem

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp)),

where :math:`\Ffun` is a vector-valued, user-provided blackbox function mapping
from :math:`\R^{\np}` to :math:`\R^{\nd}`, and :math:`\hfun` is a known,
possibly nonsmooth function mapping from :math:`\R^{\nd}` to :math:`\R`.

GOOMBAH stands for *Glassbox Optimization Of a Model of a Blackbox in a
Hypersphere*. At each iteration, GOOMBAH constructs a local model
:math:`\Mfun` of the blackbox mapping :math:`\Ffun` and approximately solves the
trust-region subproblem

.. math::

    \min_{\psp \in \Omega_k} \hfun(\Mfun(\psp)),

where :math:`\Omega_k` is the current trust-region, possibly intersected with
bound constraints. In this sense, GOOMBAH is a superset of both |pounders| and
|manifold_sampling|: when a problem can be handled by either method, it can also
be addressed by GOOMBAH through the more general subproblem
:math:`\hfun(\Mfun(\psp))`.

The GOOMBAH trust-region subproblem can be significantly harder than the
corresponding |pounders| or |manifold_sampling| subproblems. For general
nonsmooth :math:`\hfun` and nonlinear models :math:`\Mfun`, solving this
subproblem to tight tolerances may require a global optimization solver, and
individual solves can take minutes or hours. Thus, users should ensure that
the time spent solving the trust-region subproblem is commensurate with the
cost of evaluating :math:`\Ffun`.

To preserve robustness, GOOMBAH will revert to a manifold-sampling
iteration whenever the proposed trust-region subproblem solution does not
produce sufficient objective decrease. This fallback to manifold sampling,
a method that iteratively only identifies a local descent step,
allows GOOMBAH to retain the convergence properties of manifold sampling 
while exploiting more ambitious steps exploiting :math:`\hfun`
whenever they are demonstrably useful.

Two Matlab variants are provided. ``goombah`` includes recourse to the
manifold-sampling procedure when needed, while ``goombah_wo_msp`` omits this
fallback and instead proceeds using only GOOMBAH logic.

The Matlab implementation includes calls to GAMS for solving GOOMBAH
subproblems for particular examples of :math:`\hfun`. 

However, performance in the examples may depend on licensed
optimization software, such as BARON, for solving these trust-region
subproblems effectively.

.. note::
    This code does not have a test suite and is not actively tested for
    correctness.

Programmatic Interface
----------------------

|matlab|
^^^^^^^^
.. mat:autofunction:: goombah.m.goombah
