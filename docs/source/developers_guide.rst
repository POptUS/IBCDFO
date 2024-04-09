*****************
Developer's Guide
*****************

============
Contributing
============

Contributions of source code, documentation, and fixes are happily accepted
via GitHub pull request to

    https://github.com/poptus/IBCDFO/tree/develop

If you are planning a contribution, reporting bugs, or suggesting features, we
encourage you to discuss the concept by opening a GitHub issue at

  https://github.com/poptus/IBCDFO/issues

or by emailing  ``poptus@mcs.anl.gov`` and interacting with us to ensure that
your effort is well-directed.

Contribution Process
--------------------

Contrbutors should typically branch from, and make pull requests to, the
``main`` branch. The ``main`` branch is used only for releases. Pull requests
may be made from a fork, for those without repository write access.

Issues can be raised at

    https://github.com/poptus/IBCDFO/issues

Issues may include reporting bugs or suggested features.

By convention, user branch names should have a ``<type>/<name>`` format, where
example types are ``feature``, ``bugfix``, ``testing``, ``docs``, and
``experimental``.  Administrators may take a ``hotfix`` branch from the main,
which will be merged into ``main``.

When a branch closes a related issue, the pull request message should include
the phrase "Closes #N," where N is the issue number.

New features should be accompanied by at least one test case.

All pull requests to ``main`` must be reviewed by at least one administrator.

Developer's Certificate
-----------------------
.. _LICENSE: https://github.com/poptus/IBCDFO/blob/main/LICENSE

|ibcdfo| is distributed under a default MIT license (see LICENSE_), with
exceptions noted in any top-level directory that contains a separate license
file.  The act of submitting a pull request or patch will be understood as an
affirmation of the following:

.. code-block:: none

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.

=============================
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

============================
MATLAB Developer Environment
============================

**TODO**: Write this.

===================================
Adding a New Subpackage to |ibcdfo|
===================================

.. _ibcdfo_pypkg: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg
.. _VERSION: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/VERSION
.. _README.md: https://github.com/poptus/IBCDFO/blob/main/README.md
.. _tox.ini: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/tox.ini
.. _setup.py: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/setup.py
.. _load_tests.py: https://github.com/poptus/IBCDFO/blob/main/ibcdfo_pypkg/src/ibcdfo/load_tests.py

* Add new subpackage to the root of the repo in accord with the POptUS
  repository requirements
* Increment VERSION_ appropriately
* Add in the new subpackage implementation as symlinks in the correct
  ibcdfo_pypkg_ subdirectory
* Update load_tests.py_ in the main package so that it builds a suite that
  includes the tests of the subpackage
* Update the README.md_ file if necessary
* Adapt setup.py_

  * Update or expand all requirements as needed
  * Add test and package data in new subpackage to ``package_data`` if any
  * Update all other metadata as needed

* Update tox.ini_

  * Add a new testenv in tox.ini_ dedicated to the new subpackage if so
    desired

* Do local testing with |tox| if so desired
* Synchronize python version information in GitHub CI actions to version changes
  made in setup.py_ (if any)
* Commit, push, and check associated GitHub CI action logs to see if constructed
  and integrated correctly
