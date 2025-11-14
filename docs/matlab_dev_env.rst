|matlab| Developer Environment
==============================
Developers will typically work in the local clone setup as suggested for users
in the User Guide.  This includes ensuring that the |minq| submodule is
up-to-date.

Setting up and running general testing on the full |matlab| contents should also
be done as detailed in the User Guide, which includes ensuring that |bendfo| is
up-to-date.

Full Code Coverage
------------------
The script ``test_ibcdfo.m`` runs the full test suite with coverage enabled.
The test output indicates where the HTML-format code coverage report can be
found.

Low-level Testing
-----------------
To run tests of a single |matlab|-based method at a lower level, add

.. code:: console

    /path/to/BenDFO/data
    /path/to/BenDFO/m

to the |matlab| path.

To test the method without measuring coverage,

   1. change to the ``<method>/m/tests`` directory
   2. open |matlab|, and
   3. execute ``runtests`` from the prompt.

To test with coverage enabled,

   1. change to the ``<method>/m`` directory
   2. open |matlab|, and
   3. execute ``runtests("IncludeSubfolders", true, "ReportCoverageFor", pwd)``

The test output indicates where the HTML-format code coverage report can be
found.

Users can also run each test function individually with ``runtests`` as usual if
so desired.  Please refer to the inline documentation of each test or test case
for more information on how to run the test.
