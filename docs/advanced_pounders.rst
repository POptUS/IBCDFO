Advanced |pounders| Interface
=============================

Trust-region subproblem solver
------------------------------
For both Python and |matlab|, Arnold Neumaier’s minq5 solver is used by default
in |pounders| to solve trust-region subproblems.

While ``ibcdfo.pounders.pounders.pounders`` allows users to provide their own
subproblem solver, |ibcdfo| also officially provides several solvers |via| the
``create_trsp_solver`` function documented below.  Users who wish to provide
their own solver should refer to the same documentation to understand
|pounders|' interface requirements.  In addition, TRSP solvers should not be
passed to |pounders| if they alter the contents of the arguments provided to
them.

Python (TRSP solver)
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: ibcdfo.pounders.create_trsp_solver

|matlab| (TRSP solver)
^^^^^^^^^^^^^^^^^^^^^^
.. mat:autofunction:: pounders.m.create_trsp_solver

Batched ``Ffun``
----------------
By default, |pounders| (Python) calls ``Ffun`` once per point, passing a 1D
:math:`\np`-element array and expecting an :math:`\nd`-element array in
return. Setting ``Options['batched_Ffun'] = True`` instead allows ``Ffun`` to
be called with a ``(batch_size, n)`` NumPy array whose rows are points to
evaluate; ``Ffun`` must then return a ``(batch_size, m)`` NumPy array with
corresponding values in row order. This allows a user-provided ``Ffun`` to
evaluate the batch of points concurrently (if they so desire). This is a
Python-only feature.

High-level interface
--------------------
The following is a prototype for a high-level user interface for
|pounders|. Since its interface is minimal and contains only the arguments
most users must or would likely supply, it could replace
``ibcdfo.run_pounders``.  In that case, the low-level interface
``ibcdfo.pounders.pounders.pounders`` would be left in the public interface for
power users.

.. autofunction:: ibcdfo.pounders._run_user_friendly.run_user_friendly
