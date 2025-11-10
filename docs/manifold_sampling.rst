Manifold Sampling
=================

This code solves the problem

.. math::
    \min_{\psp \in \R^{\np}} h(F(\psp))

where :math:`F` is a blackbox function mapping from :math:`\R^{\np}` to
:math:`\R^{\nd}`, and :math:`h` is a nonsmooth function mapping from
:math:`\R^{\nd}` to :math:`\R`.

More details can be found in :cite:t:`MSP_2021`, :cite:t:`MSP_2018`, and
:cite:t:`MSP_2016`.

Python
------
.. todo::

    * Add Python Public API docs
    * Add testing info for this subpackage?

|matlab|
--------
.. todo::

    * Add MATLAB public API docs

Testing
^^^^^^^
.. _BenDFO: https://github.com/POptUS/BenDFO

.. todo::
    * Does this really need to be here?  Is there a valid use case for general
      users to only test just this method?  To avoid having to install minq and
      carrying out extra steps for testing POUNDerS, which user might not be
      using?  If so, do they need to run with coverage or should that be moved
      to a dev guide if valuable?

To run tests of |matlab|-based manifold sampling, in addition to general
installation steps users must have an up-to-date `BenDFO`_ clone installed and
add

.. code:: console

    /path/to/BenDFO/data
    /path/to/BenDFO/m

to their MATLAB path.

Note that some code in manifold sampling and its tests automatically alter the
|matlab| path.  While the manifold sampling tests will reset the path to its
original state if all tests pass, the path might remain altered if a test fails.

The |matlab| implementation of manifold sampling contains a single test case
``Testmanifoldsampling.m``, which calls individual tests such as
``test_one_norm.m``.

To fully test the |matlab| implementation of manifold sampling with
``Testmanifoldsampling`` but without coverage:

   1. change to the ``manifold_sampling/m/tests`` directory
   2. open |matlab|, and
   3. execute ``runtests`` from the prompt.

To fully test the |matlab| implementation of manifold sampling with
``Testmanifoldsampling`` and with coverage:

   1. change to the ``manifold_sampling/m`` directory
   2. open |matlab|, and
   3. execute ``runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)``

The test output indicates where the HTML-format code coverage report can be found.

Users can also run each test function individually as usual if so desired.
Please refer to the inline documentation of each test or test case for more
information on how to run the test.
