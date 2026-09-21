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

Model-building point evaluator
-------------------------------
By default, |pounders| (Python) calls ``Ffun`` once for each new
model-building point needed to complete its initial interpolation set.
Also, ``ibcdfo.pounders.pounders.pounders`` allows users to provide their own
model-building point evaluator. |ibcdfo| provides the ``create_mbp_evaluator``
function, which calls ``Ffun`` once with all new model-building points stacked
as rows of a single NumPy array. This allows a user-provided ``Ffun`` to
evaluate them concurrently (if they so desire). This is a Python-only feature.

The ``create_mbp_evaluator`` function is documented below.

Python (MBP evaluator)
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: ibcdfo.pounders.create_mbp_evaluator

High-level interface
--------------------
The following is a prototype for a high-level user interface for
|pounders|. Since its interface is minimal and contains only the arguments
most users must or would likely supply, it could replace
``ibcdfo.run_pounders``.  In that case, the low-level interface
``ibcdfo.pounders.pounders.pounders`` would be left in the public interface for
power users.

.. autofunction:: ibcdfo.pounders._run_user_friendly.run_user_friendly
