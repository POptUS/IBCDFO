|goombah|
=========
General Description
-------------------
This code solves the problem

.. math::
    \min_{\psp \in \R^{\np}} \hfun(\Ffun(\psp))

where :math:`\Ffun` is a vector-valued, user-provided blackbox function mapping
from :math:`\R^{\np}` to :math:`\R^{\nd}`, and :math:`\hfun` is a nonsmooth
function mapping from :math:`\R^{\nd}` to :math:`\R`.

.. todo::
    Write this!

.. note::
    This code does not have a test suite and is not actively tested for
    correctness.

Programmatic Interface
----------------------

|matlab|
^^^^^^^^
.. mat:autofunction:: goombah.m.goombah
