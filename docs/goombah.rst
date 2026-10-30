|goombah|
=========
General Description
-------------------
The *Glassbox Optimization Of a Model of a Blackbox in A Hypersphere* method,
better known as |goombah|, solves the optimization problem

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp))

subject to

.. math::
    L_j \leq \pspcomp{j} \le U_j, j=1,...,\np,

where

* :math:`\Ffun : \R^{\np} \to \R^{\nd}` is a vector-valued, user-provided blackbox function,
* :math:`\hfun : \R^{\nd} \to \R` is a user-provided, possibly nonsmooth function,
* :math:`L` is a user-provided boundary constraint that permits values of
  :math:`-\infty` to specify that the problem is unconstrained for the
  associated parameter, and
* :math:`U` is a user-provided boundary constraint that permits values of
  :math:`+\infty` to specify that the problem is unconstrained for the
  associated parameter.

|goombah| will never evaluate :math:`\Ffun` outside of the provided bounds.

At each iteration, |goombah| constructs a local model :math:`\Mfun` from
quadratic models of each component of the blackbox mapping :math:`\Ffun` and
approximates the solution to the trust-region subproblem (TRSP)

.. math::

    \min_{\psp \in \Omega_k} \hfun(\Mfun(\psp)),

where :math:`\Omega_k` is the current trust region, possibly intersected with
bound constraints. In this sense, |goombah| is a superset of both
:ref:`pounders_section` and :ref:`msp_section`: when a problem can be handled by
one of these methods, it can also be addressed by |goombah| through the more general
subproblem :math:`\hfun(\Mfun(\psp))`.

Solving the |goombah| trust-region subproblem can be significantly harder than solving the
corresponding |pounders| or |manifold_sampling| subproblems. For general
nonsmooth :math:`\hfun` and nonlinear models :math:`\Mfun`, solving this
subproblem to tight tolerances may require a global optimization solver, and
individual solves can take minutes or hours. Thus, users should ensure that
the time spent solving the trust-region subproblem is commensurate with the
cost of evaluating :math:`\Ffun`.

To preserve robustness, |goombah| reverts to a manifold sampling iteration
(see **subprob_switch** below) whenever the proposed trust-region subproblem solution does not produce
sufficient objective decrease. This fallback to manifold sampling, which
identifies only a local descent step, allows |goombah| to retain the convergence
properties of manifold sampling while otherwise potentially taking more
ambitious steps whenever they are demonstrably useful.

For algorithmic details and numerical results, see :cite:t:`MSP_2024`.

|goombah| does not prescribe how its trust-region subproblems are solved. 
Any high-quality global solver for the subproblem can be used, and the performance
of |goombah| will generally change whenever the solver is changed.
The |matlab| implementation provides two ways to solve the subproblems, which
are selected with the ``solver`` field of ``TRSP_options``:

* ``'GAMS'`` solves |goombah| trust-region subproblems by calling
  `GAMS <https://www.gams.com/>`__, which requires a local, licensed
  GAMS installation. The ``gams`` executable must be on the system path and the
  GDXMRW interface to GAMS (|eg| the ``wgdx`` and ``rgdx`` functions) must be on
  the |matlab| path. Recent GAMS releases may not include GDXMRW; if yours does
  not, it must be obtained separately (|eg| from an older GAMS
  release). Other GAMS interfaces to |matlab| are not currently supported. GAMS
  models for several choices of :math:`\hfun` are
  provided; see the description of ``TRSP_options`` below. 
  When available, we recommend the 
  `BARON <https://www.gams.com/latest/docs/S_BARON.html>`__
  optimization software.
* ``'lhs'`` evaluates :math:`\hfun(\Mfun(\psp))` on a Latin hypercube
  sampling of the trust region and then runs the |matlab| Optimization Toolbox's
  ``fmincon`` solver from the best sample point. While it requires no GAMS
  installation, it does not exploit the composite structure of the subproblem
  and may produce lower-quality TRSP solutions.  Since |goombah| performance
  will generally degrade given a lower-quality global solver, this is typically
  used only for testing and development work.

.. note::
    The |matlab| implementation's test suite is integrated into the |ibcdfo|
    continuous integration system.  However, because GAMS is not installed in
    GitHub action runners, the suite only tests the ``'lhs'`` solver (`Issue 332
    <https://github.com/POptUS/IBCDFO/issues/332>`__).

Programmatic Interface
----------------------

The interface requires that users pass in a function handle to a MATLAB function
that computes :math:`h`.  When GAMS is employed, users must also set in
**TRSP_Options.file** the path of a GAMS model file that is matched to
their **hfun** argument.  Model files for several choices of :math:`\hfun` are
provided as ``minimize_*.gms`` files in 
`goombah/m/subproblems <https://github.com/POptUS/IBCDFO/tree/main/goombah/m/subproblems>`_.

The model files

* ``minimize_censored_L1_loss_quadratic_models.gms`` and
* ``minimize_max_quadratic_mapping_of_quadratic_models.gms``

correspond to parameterized :math:`h` functions that require that users first
create their actual **hfun** argument using, respectively, the MATLAB versions
of

* :py:func:`ibcdfo.manifold_sampling.create_censored_L1_loss_hfun` and
* :py:func:`ibcdfo.manifold_sampling.create_piecewise_quadratic_hfun`

located in ``manifold_sampling/m/general_nonsmooth_h_funs``.  In addition they
must use **TRSP_Options.hfun_data** to pass to |goombah| the exact same
**C**/**D** or **Q**/**z**/**c** parameters used to construct their **hfun**

|matlab|
^^^^^^^^
.. mat:autofunction:: goombah.m.goombah
