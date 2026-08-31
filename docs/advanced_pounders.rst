Advanced |pounders| Interface
=============================

Trust-region subproblem solver
------------------------------
For both Python and |matlab|, Arnold Neumaier’s minq5 solver is used by default
in |pounders| to solve trust-region subproblems.  This has been the default
solver for a very long time.

While ``ibcdfo.pounders.pounders.pounders`` does allow users to provide their
own subproblem solver, users can also provide one of potentially many solvers
officially provided by |ibcdfo| with the ``create_trsp_solver`` function
documented below.  Users that wish to provide their own external solver should
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
The following is a prototype of a potential high-level user interface for
|pounders|.  Since it's interface is minimal and contains only those arguments
that must be or are likely to be supplied by typical users, it could replace
``ibcdfo.run_pounders``.  In that case, the low-level interface
``ibcdfo.pounders.pounders.pounders`` would be left in the public interface for
power users.

.. autofunction:: ibcdfo.pounders._run_user_friendly.run_user_friendly
