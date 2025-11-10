Python Developer Environments
=============================
Developers that prefer to work with a self-managed Python environment (virtual
or otherwise), can install a static version of |ibcdfo| by executing

.. code-block::

    $ cd /path/to/clone/ibcdfo_pypkg
    $ python setup.py sdist
    $ python -m pip install dist/ibcdfo-<version>.tar.gz

where ``<version>`` can be determined by reading the output of the setup
command.  An installation of |ibcdfo| that always uses the current state of the
code in the developer's |ibcdfo| clone (|ie| install in editable or developer
mode) can be made using

.. code-block::

    $ python -m pip install -e /path/to/clone/ibcdfo_pypkg

Developer's can use |tox| to manage their Python environments automatically.
In particular, |tox| is  setup within |ibcdfo| to manage independent Python
virtual environments for many individuals based on the efforts of the limited
few that setup and maintain |tox| in |ibcdfo|.  Assuming that users have
already executed ``tox -e coverage`` (see below), developers can activate a
Python virtual environment that always uses the current state of the code in
the developer's |ibcdfo| clone by executing

.. code-block::

    $ . /path/to/clone/ibcdfo_pypkg/.tox/coverage/bin/activate

Development with |tox|
------------------------
.. _tox: https://tox.wiki/en/latest/index.html
.. _webinar: https://www.youtube.com/watch?v=PrAyvH-tm8E

The Python packages in this repository and the management of coverage reports
for the full repository are managed with tox_, which can be used for CI work
among other work.  However, the same |tox| setups can be used by developers
if so desired, which can be useful since |tox| will automatically setup and
manage dedicated virtual environments for the developer.  The following guide
can be used to setup |tox| as a command line tool on an individual platform
in a dedicated, minimal virtual environment and is based on a webinar_ by
Oliver Bestwalter.

Developers that would like to use |tox| should learn about the tool so that,
at the very least, they understand the difference between running |tox| and
``tox -r``.

To create a Python virtual environment based on a desired Python dedicated to
hosting |tox|, execute some variation of

.. code-block:: console

    $ cd
    $ deactivate (to deactivate the current virtual environment if you are in one)
    $ /path/to/desired/python --version
    $ /path/to/desired/python -m venv $HOME/.toxbase
    $ ./.toxbase/bin/pip list
    $ ./.toxbase/bin/python -m pip install --upgrade pip
    $ ./.toxbase/bin/pip install --upgrade setuptools
    $ ./.toxbase/bin/pip install tox
    $ ./.toxbase/bin/tox --version
    $ ./.toxbase/bin/pip list

To avoid the need to activate ``.toxbase``, we setup |tox| in ``PATH`` for
use across all development environments that we might have on our system. In
the following, please replace ``.bash_profile`` with the appropriate shell
configuration file and tailor to your needs.

.. code-block:: console

    $ mkdir $HOME/local/bin
    $ ln -s $HOME/.toxbase/bin/tox $HOME/local/bin/tox
    $ vi $HOME/.bash_profile (add $HOME/local/bin to PATH)
    $ . $HOME/.bash_profile
    $ which tox
    $ tox --version

The following commands can be run from ``/path/to/IBCDFO/ibcdfo_pypkg``

* ``tox -r -e coverage``

  * Execute the full test suite for the package and save coverage results to the
    coverage file
  * The test runs the package code in the local clone rather than code installed
    into python so that coverage results accessed through web services such as
    Coveralls are clean and straightforward

* ``tox -r -e nocoverage``

  * Execute the full test suite for the package using the code installed into
    python

* ``tox -r -e pounders``

  * Execute the test suite for the pounders subpackage only using the code
    installed into python

* ``tox -r -e manifold_sampling``

  * Execute the test suite for the manifold_sampling subpackage only using the
    code installed into python

* ``tox -r -e report``

  * It is intended that this be run after or with coverage
  * Display a report and generate an HTML report for the package's full test
    suite

* ``tox -r -e check``

  * This is likely only useful for developers working on a local clone
  * This task will never call any tools that automatically **alter** files
  * Run several checks on the code to report possible issues

* ``tox -r -e html``

  * Generate and render |ibcdfo|'s documentation locally in HTML

* ``tox -r -e pdf``

  * Generate and render |ibcdfo|'s documentation locally as a PDF file

Additionally, you can run any combination of the above such as
``tox -r -e report,coverage``.

If the environment variable ``COVERAGE_FILE`` is set, then this is the coverage
file that will be used with all associated work.  If it is not specified, then
the coverage file is ``.coverage_ibcdfo``.

Manual Developer Testing
------------------------
It is possible to test manually outside of |tox|, which could be useful for
testing at the level of a single test.

To run the full test suite, execute

.. code-block:: console

    $ python -m unittest ibcdfo

To run only the ``TestPoundersSimple`` pounders test, execute

.. code-block:: console

    $ python -m unittest ibcdfo.pounders.tests.TestPoundersSimple
