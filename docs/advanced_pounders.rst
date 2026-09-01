Advanced |pounders| Interface
=============================

Trust-region subproblem solver
------------------------------
For both Python and |matlab|, Arnold Neumaier’s minq5 solver is used by default
in |pounders| to solve trust-region subproblems.

While ``ibcdfo.pounders.pounders.pounders`` allows users to provide their
own subproblem solver, |ibcdfo| also officially provides several solvers via the
``create_trsp_solver`` function
documented below.  Users who wish to provide their own solver should
refer to the same documentation to understand |pounders|'s interface
requirements.

Python
^^^^^^
.. autofunction:: ibcdfo.pounders.create_trsp_solver

|matlab|
^^^^^^^^
.. mat:autofunction:: pounders.m.create_trsp_solver

High-level interface
--------------------
The following is a prototype for a high-level user interface for
|pounders|. Since its interface is minimal and contains only the arguments
most users must or would likely supply, it could replace
``ibcdfo.run_pounders``.  In that case, the low-level interface
``ibcdfo.pounders.pounders.pounders`` would be left in the public interface for
power users.

.. autofunction:: ibcdfo.pounders._run_user_friendly.run_user_friendly
