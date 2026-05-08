|ibcdfo|
========
.. _POptUS: https://github.com/POptUS
.. _license: https://github.com/poptus/IBCDFO/blob/main/LICENSE
.. _Issue: https://github.com/POptUS/IBCDFO/issues

This documentation provides information for using the set of Interpolation-Based
optimization methods for Composite Derivative-Free Optimization, which are
collectively referred to as |ibcdfo| and are part of POptUS_: Practical
Optimization Using Structure. IBCDFO is a collection of interpolation-based
methods for solving structured, composite derivative-free optimization problems.

All code included in |ibcdfo| is open source, with the particular form of
license contained in the top-level subdirectories. If such a subdirectory does
not contain a LICENSE file, then it is automatically licensed as described in
the otherwise encompassing IBCDFO license_.

Support
-------
To

* report potential problems with |ibcdfo|,
* propose a change, or
* request a new feature,

please check if a related Issue_ already exists before creating a new Issue. For
all other communication, please email the |poptus| development team at

.. code-block:: none

    poptus@mcs.anl.gov

Contributing to |ibcdfo|
------------------------

Contributions are welcome in a variety of forms; please see
:numref:`contributing:Contributing` in the Developer Guide.

Cite |ibcdfo|
-------------

.. code-block:: console

  @techreport{ibcdfo,
    author = {Jeffrey Larson and Matt Menickelly and Stefan M. Wild},
    title  = {Interpolation-Based Composite Derivative-Free Optimization},
    url    = {https://github.com/POptUS/IBCDFO},
    number = {Version 0.1.0},
    year   = {2026},
    doi    = {10.11578/dc.20240627.2}
  }

.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: User Guide:

   get_started
   pounders
   manifold_sampling
   goombah
   bibliography

.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: Advanced Users:

   advanced_pounders

.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: Developer Guide:

   contributing
   python_dev_env
   matlab_dev_env
   documentation
   new_subpackage
