Python Developer Environments
=============================
.. _tox: https://tox.wiki

The |ibcdfo| repository includes a tox_ setup that defines a set of
predefined development tasks, each of which runs in a dedicated Python virtual
environment created and managed automatically by |tox|.

Development with |tox|
------------------------
.. _Developer Guide: https://poptus.readthedocs.io

To set up |tox| for use with this repository, please follow the directions
included in the |poptus| `Developer Guide`_.

The following commands can be run from ``/path/to/IBCDFO/ibcdfo_pypkg``:

* ``tox -e coverage``

  * Execute the full test suite for the package and save coverage results.
  * The ``COVERAGE_FILE`` environment variable can optionally be set to define
    the output file name. By default, results are written to
    ``.coverage_ibcdfo``.
  * Tests run against the package as installed in the local clone, ensuring that
    coverage results (e.g., for CodeCov) are clean and straightforward. This
    mimics an editable (developer) installation.

* ``tox -e nocoverage``

  * Execute the full test suite using the code installed into a Python virtual
    environment. This mimics a standard installation from a package
    distribution (e.g., a wheel).

* ``tox -e pounders``

  * Execute the test suite for the |pounders| subpackage using the installed
    package in a virtual environment.

* ``tox -e manifold_sampling``

  * Execute the test suite for the Manifold Sampling subpackage using the
    installed package in a virtual environment.

* ``tox -e report``

  * Intended to be run after (or alongside) ``coverage``.
  * Display a coverage report and generate XML and HTML outputs.
  * The ``COVERAGE_XML_FILE`` and ``COVERAGE_HTML_FILE`` environment variables
    can optionally be set to define output file names. Defaults are
    ``cobertura_ibcdfo.xml`` and ``htmlcov_ibcdfo``.

* ``tox -e check``

  * Run code quality checks to report potential issues. The codebase satisfies
    |ibcdfo| coding standards if this and the ``format`` task both pass.

* ``tox -e format``

  * **NOTE: This may modify Python code in your local clone.**
  * Automatically reformat code using the ``black`` tool. The codebase satisfies
    |ibcdfo| coding standards after applying this task, provided ``check`` still
    passes.

* ``tox -e format_safe``

  * Report changes that would be made by the ``black`` tool without modifying
    files.

* ``tox -e html``

  * Generate and render |ibcdfo| documentation locally in HTML format.
  * Documentation is built from the local clone rather than an installed
    package, enabling fast, interactive documentation work.

..
    * ``tox -e pdf``
    
      * Generate and render |ibcdfo| documentation locally as a PDF.
      * Documentation is built from the local clone rather than an installed
        package.
      * This task uses ``make`` and requires a LaTeX installation.

Additionally, you can run multiple tasks together, such as
``tox -e report,coverage``.

Each task can be run as either ``tox -e <task>`` or ``tox -r -e <task>``.
Developers are responsible for choosing the appropriate option for their
situation.

Direct use of |tox| venvs
-------------------------
Developers may directly use the virtual environments (venvs) created and
managed by |tox|. For example, the venvs created for the ``coverage`` and
``html`` tasks can be especially useful, since |ibcdfo| is installed in
editable mode, facilitating interactive development and testing.

To run the full test suite during an interactive debugging session, developers
can create a clean ``coverage`` venv and activate it:

.. code:: console

    $ cd /path/to/IBCDFO/ibcdfo_pypkg
    $ tox -r -e coverage
    $ . ./.tox/coverage/bin/activate

Then run the full test suite:

.. code:: console

    $ python -m unittest ibcdfo

For example, after modifying |pounders| code or the
``TestPoundersSimple`` test, a developer can rerun just that test:

.. code:: console

    $ python -m unittest ibcdfo.pounders.tests.TestPoundersSimple

to quickly check the effect.
