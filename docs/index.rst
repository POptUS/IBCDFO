|ibcdfo|
========
.. _POptUS: https://github.com/POptUS
.. _license: https://github.com/poptus/IBCDFO/blob/main/LICENSE

This document provides information for using the set of Interpolation-Based
optimization methods for Composite Derivative-Free Optimization, which are
collectively referred to as |ibcdfo| and are part of POptUS_: Practical
Optimization Using Structure.

Please refer to :cite:t:`IBCDFO_2024`, :cite:t:`POUNDERS_TAO_2017`,
:cite:t:`MSP_2021`, :cite:t:`MSP_2018`, and :cite:t:`MSP_2016` for detailed
information regarding the different methods provided by |ibcdfo|.

All code included in IBCDFO is open source, with the particular form of license
contained in the top-level subdirectories.  If such a subdirectory does not
contain a LICENSE file, then it is automatically licensed as described in the
otherwise encompassing IBCDFO license_.

Support
-------
TODO: Statement about use of Issues?

To seek support or report issues, e-mail

.. code-block:: none

    poptus@mcs.anl.gov

Cite |ibcdfo|
-------------

.. code-block:: console

  @misc{ibcdfo,
    author = {Jeffrey Larson and Matt Menickelly and Jared P. O'Neal and Stefan M. Wild},
    title  = {Interpolation-Based Composite Derivative-Free Optimization},
    url    = {https://github.com/POptUS/IBCDFO},
    year   = {2024},
  }

Contributing to |ibcdfo|
------------------------

Contributions are welcome in a variety of forms; please see
:numref:`contributing:Contributing`.

.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: User Guide:

   get_started
   pounders
   manifold_sampling
   bibliography

.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: Developer Guide:

   contributing
   python_dev_env
   matlab_dev_env
   documentation
   new_subpackage
