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

|goombah| stands for *Glassbox Optimization Of a Model of a Blackbox in a
Hypersphere*. At each iteration, |goombah| constructs a local model
:math:`\Mfun` of the blackbox mapping :math:`\Ffun` and approximately solves the
trust-region subproblem

.. math::

    \min_{\psp \in \Omega_k} \hfun(\Mfun(\psp)),

where :math:`\Omega_k` is the current trust-region, possibly intersected with
bound constraints. In this sense, |goombah| is a superset of both |pounders| and
|manifold_sampling|: when a problem can be handled by either method, it can also
be addressed by |goombah| through the more general subproblem
:math:`\hfun(\Mfun(\psp))`.

The |goombah| trust-region subproblem can be significantly harder than the
corresponding |pounders| or |manifold_sampling| subproblems. For general
nonsmooth :math:`\hfun` and nonlinear models :math:`\Mfun`, solving this
subproblem to tight tolerances may require a global optimization solver, and
individual solves can take minutes or hours. Thus, users should ensure that
the time spent solving the trust-region subproblem is commensurate with the
cost of evaluating :math:`\Ffun`.

To preserve robustness, |goombah| reverts to a manifold-sampling
iteration whenever the proposed trust-region subproblem solution does not
produce sufficient objective decrease. This fallback to manifold sampling
(which identifies only a local descent step)
allows |goombah| to retain the convergence properties of manifold sampling
while taking more ambitious steps exploiting :math:`\hfun`
whenever they are demonstrably useful.

For algorithmic details and numerical results, see :cite:t:`MSP_2024`.

|goombah| does not prescribe how its trust-region subproblem is solved. 
Any high-quality global solver for the subproblem can be used, and the performance
of |goombah| will change with the quality of that solver. 
The |matlab| implementation provides two ways to solve the subproblems, which
are selected with the ``solver`` field of ``TRSP_options``:

* ``'GAMS'`` solves |goombah| trust-region subproblems by calling
  `GAMS <https://www.gams.com/>`__, so using it requires an existing, licensed
  GAMS installation. The ``gams`` executable must be on the system path and the
  GDXMRW interface to GAMS (and the ``wgdx`` and ``rgdx`` functions) must be on
  the MATLAB path. Recent GAMS releases may not include GDXMRW; if yours does
  not, it must be obtained separately (for example, from an older GAMS
  release). Other GAMS interfaces to MATLAB are not currently supported. GAMS
  models for several choices of :math:`\hfun` are
  provided; see the description of ``TRSP_options`` below. 
  If possible, we recommend the 
  `BARON <https://www.gams.com/latest/docs/S_BARON.html>`__
  optimization software.
* ``'lhs'`` evaluates :math:`\hfun(\Mfun(\psp))` at a large Latin hypercube
  sample of the trust region and then runs ``fmincon`` (from the MATLAB
  Optimization Toolbox) from the best sample point. It requires no GAMS
  installation, but it does not exploit the composite structure of the
  subproblem and may produce lower-quality TRSP solution. GOOBMAH performance
  will degrade relative to a higher-quality global solver. 
  We currently issue a warning when this solver is used.

.. note::
    The |matlab| implementation has a test suite that runs on every pull
    request. Because GAMS is not available in that testing environment, the
    tests use the ``'lhs'`` solver and do not exercise the GAMS-based
    subproblem solves.

Programmatic Interface
----------------------

|matlab|
^^^^^^^^
.. mat:autofunction:: goombah.m.goombah
