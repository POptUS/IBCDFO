Python Developer Environments
=============================
.. _tox: https://tox.wiki

The |ibcdfo| repository includes a tox_ setup that defines different
predefined development tasks, each of which is run in a dedicated Python virtual
environment created and managed automatically by |tox|.

Development with |tox|
------------------------
.. _Developer Guide: https://poptus.readthedocs.io

To setup |tox| for use with this repository, please follow the directions
included in the |poptus| `Developer Guide`_.

The following commands can be run from ``/path/to/IBCDFO/ibcdfo_pypkg``

* ``tox -e coverage``

  * Execute the full test suite for the package and save coverage results to the
    coverage file
  * The ``COVERAGE_FILE`` environment variable can optionally be set to define
    the name of the file that coverage results will be written to.  By default,
    results are written to ``.coverage_ibcdfo``.
  * The test runs the package code in the local clone rather than code installed
    into Python so that coverage results accessed through web services such as
    CodeCov are clean and straightforward

* ``tox -e nocoverage``

  * Execute the full test suite for the package using the code installed into
    Python

* ``tox -e pounders``

  * Execute the test suite for the |pounders| subpackage using only the code
    installed into Python

* ``tox -e manifold_sampling``

  * Execute the test suite for the Manifold Sampling subpackage using only the
    code installed into Python

* ``tox -e report``

  * It is intended that this be run after or with ``coverage``
  * Display a code coverage report for the package's full test suite and
    generate XML and HTML versions of the report.
  * The ``COVERAGE_XML_FILE`` and ``COVERAGE_HTML_FILE`` environment variables
    can optionally be set to define the name of the XML- and HTML-format
    reports.  Default report names are ``cobertura_ibcdfo.xml`` and
    ``htmlcov_ibcdfo``.

* ``tox -e check``

  * Run several checks on the code to report possible issues

* ``tox -e format``

  * **NOTE: This will potentially alter Python code in your local clone.**
  * Automatically reformat Python code in the package based on the ``black``
    tool's criteria.

* ``tox -e format_safe``

  * Report all changes that would be made to Python code in the package to
    satisfy the ``black`` tool's criteria.

* ``tox -e html``

  * Generate and render |ibcdfo|'s documentation locally in HTML
  * Documentation is built from the code in the local clone rather than code
    installed into Python to support quick, interactive documentation work

..
    * ``tox -e pdf``
    
      * Generate and render |ibcdfo|'s documentation locally as a PDF file
      * Documentation is built from the code in the local clone rather than code
        installed into Python.
      * This task uses ``make`` and requires a LaTeX installation.

Additionally, you can run any combination of the above such as
``tox -e report,coverage``.

Note that each task can be run as ``tox -r -e <task>`` or ``tox -e <task>``.
Developers are responsible for determining which is correct for their current
situation.

Direct use of |tox| venvs
-------------------------
Developers are free to use the virtual environments created and managed
automatically by |tox|.  The venvs created for executing the ``coverage`` and
``html`` tasks, for instance, can be especially useful since |ibcdfo| is
installed in editable mode for these tasks, which facilitates interactive
development and testing of the Python code and its documentation.

To run the full test suite during an interactive debugging session, developers
can create a clean version of the ``coverage`` venv and activate it for
immediate use with

.. code:: console

    $ cd /path/to/IBCDFO/ibcdfo_pypkg
    $ tox -r -e coverage
    $ . ./.tox/coverage/bin/activate

and subsequently run the full test suite in the venv with

.. code:: console

    $ python -m unittest ibcdfo

A user could then alter |pounders| code or its ``TestPoundersSimple`` test and
rerun just that test in the venv with

.. code:: console

    $ python -m unittest ibcdfo.pounders.tests.TestPoundersSimple
