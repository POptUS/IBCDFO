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

Programmatic Interface
----------------------

|matlab|
^^^^^^^^
.. mat:autofunction:: goombah.m.goombah
